#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

int NUM_BODIES;
int STEPS;
#define G 1.0
#define DT 0.01
#define SOFTENING 1e-2

typedef struct {
    double x, y;
    double vx, vy;
    double mass;
} Body;


void initialize_bodies(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].x = (double)rand() / RAND_MAX;
        bodies[i].y = (double)rand() / RAND_MAX;
        bodies[i].vx = 0.0;
        bodies[i].vy = 0.0;
        bodies[i].mass = 1.0;
    }
}

//Força entre vetor 'local' e vetor 'remote'
void compute_pair_forces(Body *local, Body *remote, int n_local, int n_remote,
                         double *fx, double *fy) {
    for (int i = 0; i < n_local; i++) {
        for (int j = 0; j < n_remote; j++) {
            double dx = remote[j].x - local[i].x;
            double dy = remote[j].y - local[i].y;
            double dist_sq = dx * dx + dy * dy + SOFTENING * SOFTENING;
            
            double inv_dist = 1.0 / sqrt(dist_sq);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;
            double f = G * local[i].mass * remote[j].mass * inv_dist3;

            fx[i] += f * dx;
            fy[i] += f * dy;
        }
    }
}

void update_positions(Body *b, double *fx, double *fy, int n) {
    for (int i = 0; i < n; i++) {
        double ax = fx[i] / b[i].mass;
        double ay = fy[i] / b[i].mass;
        b[i].vx += ax * DT;
        b[i].vy += ay * DT;
        b[i].x += b[i].vx * DT;
        b[i].y += b[i].vy * DT;
    }
}

// --- Versão Sequencial (para validação ou size=1) ---
void sequencial(int n) {
    printf("Modo: SEQUENCIAL (1 processo)\n");
    Body *bodies = malloc(n * sizeof(Body));
    double *fx = malloc(n * sizeof(double));
    double *fy = malloc(n * sizeof(double));
    
    initialize_bodies(bodies, n);

    for (int s = 0; s < STEPS; s++) {
        for(int i=0; i<n; i++) fx[i] = fy[i] = 0.0;
        compute_pair_forces(bodies, bodies, n, n, fx, fy);
        update_positions(bodies, fx, fy, n);
    }

    printf("Simulação Sequencial Concluída. Posição corpo 0: %.3f\n", bodies[0].x);
    free(bodies); free(fx); free(fy);
}


int main(int argc, char **argv) {

    if(argc != 3) {
        printf("Uso: ./%s <num_corpos> <num_passos>\n", argv[0]);
        return 1;
    }

    NUM_BODIES = atoi(argv[1]);
    STEPS = atoi(argv[2]);

    double start_time, end_time;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verificação de divisibilidade básica
    if (NUM_BODIES % size != 0) {
        if (rank == 0) printf("Erro: NUM_BODIES (%d) deve ser divisível por Size (%d)\n", NUM_BODIES, size);
        MPI_Finalize();
        return 1;
    }

    //
    //  Sequencial
    //
    if (size == 1) {
        start_time = MPI_Wtime();
        sequencial(NUM_BODIES);
        end_time = MPI_Wtime();
        printf("Tempo Total: %f s\n", end_time - start_time);
        MPI_Finalize();
        return 0;
    }
    

    //
    // Paralelo
    //
    int n_local = NUM_BODIES / size;
    int prev = (rank - 1 + size) % size;
    int next = (rank + 1) % size;

    Body *local_bodies = malloc(n_local * sizeof(Body));
    double *fx = calloc(n_local, sizeof(double));
    double *fy = calloc(n_local, sizeof(double));

    // Buffers para o Anel (Double Buffering para SendRecv seguro)
    // remote_buf_A: armazena os dados que acabei de receber (ou meus próprios no inicio)
    // remote_buf_B: armazena os dados que estou recebendo agora
    Body *remote_buf_A = malloc(n_local * sizeof(Body));
    Body *remote_buf_B = malloc(n_local * sizeof(Body));

    start_time = MPI_Wtime();

    // Distribuição Inicial (Scatter)
    if (rank == 0) {
        Body *all_bodies = malloc(NUM_BODIES * sizeof(Body));
        initialize_bodies(all_bodies, NUM_BODIES);
        MPI_Scatter(all_bodies, n_local * sizeof(Body), MPI_BYTE,
                    local_bodies, n_local * sizeof(Body), MPI_BYTE,
                    0, MPI_COMM_WORLD);
        free(all_bodies);
    } else {
        MPI_Scatter(NULL, n_local * sizeof(Body), MPI_BYTE,
                    local_bodies, n_local * sizeof(Body), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    // 3. Loop de Tempo
    for (int step = 0; step < STEPS; step++) {
        
        // Zera forças
        for (int i = 0; i < n_local; i++) fx[i] = fy[i] = 0.0;

        // Preparação para o Anel
        // Na fase 0, calculo forças com meus próprios corpos.
        // Copiamos local -> remote_buf_A para iniciar o ciclo de envios
        // (Isso gasta um pouco de memória, mas simplifica a lógica de ponteiros do anel)
        for(int i=0; i<n_local; i++) remote_buf_A[i] = local_bodies[i];

        Body *send_ptr = remote_buf_A; // O que vou enviar (e usar para cálculo atual)
        Body *recv_ptr = remote_buf_B; // Onde vou receber o próximo

        // 4. Fases Paralelas (Ring Shift)
        for (int phase = 0; phase < size; phase++) {
            //Fase 1: processamento local
            // Computação: Meus corpos (estáticos) vs Corpos "Viajantes" (send_ptr)
            compute_pair_forces(local_bodies, send_ptr, n_local, n_local, fx, fy);

            //Fase 2: parada
            // Comunicação: Se não for a última fase, circular os dados
            if (phase < size - 1) {

                //Fase 3 troca com vizinhos
                MPI_Sendrecv(send_ptr, n_local * sizeof(Body), MPI_BYTE, next, 0,
                             recv_ptr, n_local * sizeof(Body), MPI_BYTE, prev, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Swap de ponteiros:
                // O que recebi (recv_ptr) vira o dado para calcular e enviar na próxima (send_ptr)
                Body *temp = send_ptr;
                send_ptr = recv_ptr;
                recv_ptr = temp; 
                // Agora recv_ptr aponta para o buffer antigo, pronto para ser sobrescrito na próxima recepção
            }
        }

        // Integração
        update_positions(local_bodies, fx, fy, n_local);
    }

    end_time = MPI_Wtime();
    
    // Coleta final para impressão (Opcional, apenas corpo 0)
    if (rank == 0) {
        printf("Paralelo (%d procs): Tempo = %f s\n", size, end_time - start_time);
        printf("Body 0 Pos: %.3f, %.3f\n", local_bodies[0].x, local_bodies[0].y);
    }

    free(local_bodies);
    free(remote_buf_A);
    free(remote_buf_B);
    free(fx);
    free(fy);

    MPI_Finalize();
    return 0;
}
