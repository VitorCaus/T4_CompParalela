#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
// #include "types.h"

int NUM_BODIES = 250;
#define G 1.0
#define DT 0.01
int STEPS = 50000;
#define SOFTENING 1e-2

typedef struct {
    double x, y;
    double vx, vy;
    double mass;
} Body;

void initialize_bodies(Body *bodies)
{
  for (int i = 0; i < NUM_BODIES; i++)
  {
    bodies[i].x = (double)rand() / RAND_MAX;
    bodies[i].y = (double)rand() / RAND_MAX;
    bodies[i].vx = 0.0;
    bodies[i].vy = 0.0;
    bodies[i].mass = 1.0;
  }
}

void compute_pair_forces(Body *local, Body *remote, int n_local, int n_remote,
                         double *fx, double *fy)
{
  for (int i = 0; i < n_local; i++)
  {
    for (int j = 0; j < n_remote; j++)
    {
      if (&local[i] == &remote[j])
        continue;

      double dx = remote[j].x - local[i].x;
      double dy = remote[j].y - local[i].y;
      double dist_sq = dx * dx + dy * dy + SOFTENING * SOFTENING;
      double inv = 1.0 / sqrt(dist_sq);
      double inv3 = inv * inv * inv;
      double f = G * local[i].mass * remote[j].mass * inv3;
      fx[i] += f * dx;
      fy[i] += f * dy;
    }
  }
}

void update_positions(Body *b, double *fx, double *fy, int n)
{
  for (int i = 0; i < n; i++)
  {
    double ax = fx[i] / b[i].mass;
    double ay = fy[i] / b[i].mass;
    b[i].vx += ax * DT;
    b[i].vy += ay * DT;
    b[i].x += b[i].vx * DT;
    b[i].y += b[i].vy * DT;
  }
}

int main(int argc, char **argv)
{
  if(argc != 3){
    printf("Uso: ./%s <num_corpos> <num_passos>\n", argv[0]);
    return 1;
  }

  NUM_BODIES = atoi(argv[1]);
  STEPS = atoi(argv[2]);


  MPI_Init(&argc, &argv);
  double end_time, start_time;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (NUM_BODIES% size != 0)
  {
    if (rank == 0)
      printf("NUM_BODIESprecisa ser divisível por numero de processos.\n");
    MPI_Finalize();
    return 1;
  }

  int n_local = NUM_BODIES/ size;

  Body *local = malloc(n_local * sizeof(Body));
  Body *recvA = malloc(n_local * sizeof(Body)); // buffers para receber em alternância
  Body *recvB = malloc(n_local * sizeof(Body));
  double *fx = calloc(n_local, sizeof(double));
  double *fy = calloc(n_local, sizeof(double));

  if (!local || !recvA || !recvB || !fx || !fy)
  {
    fprintf(stderr, "Erro de alocacao\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  start_time = MPI_Wtime();

  // processo 0 inicializa todos e faz scatter
  if (rank == 0)
  {
    Body *all = malloc(NUM_BODIES* sizeof(Body));
    if (!all)
    {
      fprintf(stderr, "malloc failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    initialize_bodies(all);

    MPI_Scatter(all, n_local * sizeof(Body), MPI_BYTE,
                local, n_local * sizeof(Body), MPI_BYTE,
                0, MPI_COMM_WORLD);

    free(all);
  }
  else
  {
    MPI_Scatter(NULL, n_local * sizeof(Body), MPI_BYTE,
                local, n_local * sizeof(Body), MPI_BYTE,
                0, MPI_COMM_WORLD);
  }

  int dest = (rank + 1) % size;
  int src = (rank - 1 + size) % size;

  for (int step = 0; step < STEPS; step++)
  {

    // resetar forças locais
    for (int i = 0; i < n_local; i++)
      fx[i] = fy[i] = 0.0;

    // compute interactions local-local
    compute_pair_forces(local, local, n_local, n_local, fx, fy);

    // ring: enviar bloco atual para dest e receber de src; propagar blocos em ambos os sentidos
    Body *sendbuf = local;
    Body *recvbuf = recvA;
    Body *next_recv = recvB;

    for (int iter = 0; iter < size - 1; iter++)
    {
      // envia sendbuf para dest e recebe em recvbuf de src
      MPI_Sendrecv(sendbuf, n_local * sizeof(Body), MPI_BYTE, dest, 0,
                   recvbuf, n_local * sizeof(Body), MPI_BYTE, src, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // computa forças entre local e o bloco recebido
      compute_pair_forces(local, recvbuf, n_local, n_local, fx, fy);

      // preparar próximo envio: enviar o bloco que acabei de receber
      sendbuf = recvbuf;

      // alterna buffers para a próxima recepção para evitar sobrescrever
      recvbuf = next_recv;
      next_recv = sendbuf == recvA ? recvA : recvB; // simples alternância
      // simpler way: swap pointers
      Body *tmp = next_recv;
      (void)tmp;
    }

    // atualizar posições somente dos corpos locais
    update_positions(local, fx, fy, n_local);

    // if (rank == 0 && step % 200 == 0)
      //printf("Step %d: Body0=(%.3f, %.3f)\n", step, local[0].x, local[0].y);
  }

  free(local);
  free(recvA);
  free(recvB);
  free(fx);
  free(fy);

  MPI_Finalize();
  end_time = MPI_Wtime();
  
  printf("Tempo total: %f segundos\n", end_time - start_time);
  return 0;
}

