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
            
            // Nota: Com Allgather, calcularemos a força de um corpo sobre ele mesmo (dx=0, dy=0).
            // O SOFTENING evita divisão por zero. A força resultante será 0, o que é fisicamente correto
            // (um corpo não empurra a si mesmo).
            
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
    // Paralelo (Versão Allgather)
    //
    int n_local = NUM_BODIES / size;
    
    // Alocação
    Body *local_bodies = malloc(n_local * sizeof(Body));
    double *fx = calloc(n_local, sizeof(double));
    double *fy = calloc(n_local, sizeof(double));

    // Buffer GLOBAL para receber dados de TODOS os processos
    // Diferente do Ring, aqui alocamos espaço para o universo inteiro
    Body *all_bodies = malloc(NUM_BODIES * sizeof(Body));

    start_time = MPI_Wtime();

    // Distribuição Inicial (Scatter)
    if (rank == 0) {
        // Usamos um buffer temporário apenas para inicializar e distribuir
        Body *temp_all = malloc(NUM_BODIES * sizeof(Body));
        initialize_bodies(temp_all, NUM_BODIES);
        MPI_Scatter(temp_all, n_local * sizeof(Body), MPI_BYTE,
                    local_bodies, n_local * sizeof(Body), MPI_BYTE,
                    0, MPI_COMM_WORLD);
        free(temp_all);
    } else {
        MPI_Scatter(NULL, n_local * sizeof(Body), MPI_BYTE,
                    local_bodies, n_local * sizeof(Body), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    // 3. Loop de Tempo
    for (int step = 0; step < STEPS; step++) {
        
        // Zera forças
        for (int i = 0; i < n_local; i++) fx[i] = fy[i] = 0.0;

        // --- COMUNICAÇÃO (Substitui o Loop de Fases/Ring) ---
        // Coleta os corpos locais de todos e monta o vetor 'all_bodies' completo em cada processo
        // Nota: O count de recebimento (recvcount) é a quantidade recebida de CADA processo, não o total.
        MPI_Allgather(local_bodies, n_local * sizeof(Body), MPI_BYTE,
                      all_bodies,   n_local * sizeof(Body), MPI_BYTE,
                      MPI_COMM_WORLD);

        // --- COMPUTAÇÃO ---
        // Calcula força entre meus corpos locais (n_local) e TODO o universo (NUM_BODIES)
        compute_pair_forces(local_bodies, all_bodies, n_local, NUM_BODIES, fx, fy);

        // Integração (Atualiza apenas os meus locais)
        update_positions(local_bodies, fx, fy, n_local);
    }

    end_time = MPI_Wtime();
    
    // Coleta final para impressão (Opcional, apenas corpo 0)
    if (rank == 0) {
        printf("Paralelo Allgather (%d procs): Tempo = %f s\n", size, end_time - start_time);
        printf("Body 0 Pos: %.3f, %.3f\n", local_bodies[0].x, local_bodies[0].y);
    }

    free(local_bodies);
    free(all_bodies); // Libera o buffer global
    free(fx);
    free(fy);

    MPI_Finalize();
    return 0;
}
