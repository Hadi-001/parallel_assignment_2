CC = gcc
MPICC = mpicc
CFLAGS = -O2 -std=c11 -Wall
OMP_FLAGS = -fopenmp

all: sequential openmp_rl mpi_rl

sequential: sequential.c gridworld.h
	$(CC) $(CFLAGS) -o sequential sequential.c -lm

openmp_rl: openmp_rl.c gridworld.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o openmp_rl openmp_rl.c -lm

mpi_rl: mpi_rl.c gridworld.h
	$(MPICC) $(CFLAGS) -o mpi_rl mpi_rl.c -lm

clean:
	rm -f sequential openmp_rl mpi_rl
	rm -rf output/ results/

.PHONY: all clean
