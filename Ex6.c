#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

double* Read_matrix(char* prompt, int*, int* n, int p, int my_rank);
void Print_matrix(char title[], double local_A[], int m, int n, int p, int my_rank);
void Parallel_matrix_matrix_prod(double local_A[], double local_B[], double local_C[], int n, int local_n, int local_m, int l, int p, int my_rank);
		
/*-------------------------------------------------------------------*/
main(int argc, char* argv[])
{
	int my_rank; /* rank of process */
   	int p; /* number of processes */
	double* local_A;
	double* local_B;
	double* local_C;
	int	m, local_m;
	int	n, local_n;
	int	l;
	double start, finish, loc_elapsed, elapsed;
	int i,j;

   	/* Start up MPI */
   	MPI_Init(&argc, &argv);

   	/* Find out process rank */
   	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
   	/* Find out number of processes */
   	MPI_Comm_size(MPI_COMM_WORLD, &p);

	MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0)
                istart = MPI_Wtime();

	local_A = Read_matrix("A.txt", &m, &n, p, my_rank);
	local_B = Read_matrix("B.txt", &n, &l, p, my_rank);
	local_m = m/p;
	local_n = n/p;
	local_C = malloc(local_m*l*sizeof(double));

	MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0)
                start = MPI_Wtime();
	Parallel_matrix_matrix_prod(local_A, local_B, local_C, n, local_n, local_m, l,p,my_rank);
	MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0)
                finish = MPI_Wtime();

	Print_matrix("C.txt", local_C, m, l, p, my_rank);
	
	free(local_A);
	free(local_B);
	free(local_C);

	MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0)
        {
                ifinish = MPI_Wtime();
                //printf("Time excluding I/O : %lf\n",finish-start);
                //printf("Total time including I/O : %lf\n", ifinish-istart);
                //printf("%lf\n",ifinish-istart);
		printf("%lf\n",finish-start);
        }
	

	MPI_Finalize();

} /* main */

/*****************************************************************/
double* Read_matrix(
	char*  	filename	/* in  */,
        int*	m	     	/* in  */,
	int*    n 		/* in  */,
	int    	p          	/* in  */,
        int    	my_rank    	/* in  */) 
{
	
	int i,j,q;
    	double* temp = NULL;
	double* local_A;
	FILE *ifp;
    	MPI_Status status;

	if (my_rank == 0)
	{
		ifp = fopen(filename, "r"); 
		fscanf(ifp, "%d", m);
		fscanf(ifp, "%d", n);		
	}

	MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int local_m = *m/p;

	local_A = malloc(local_m*(*n)*sizeof(double));
	temp = malloc(local_m*(*n)*sizeof(double));


	if (my_rank == 0) 
	{
		for (i = 0; i<local_m; i++)
		{
			for (j=0;j<(*n); j++)
				fscanf(ifp,"%lf", &local_A[i*(*n)+j]);
		}

        for (q = 1; q < p; q++) 
		{
			for (i = 0; i<local_m; i++)
			{
				for (j=0;j<(*n); j++)
					fscanf(ifp,"%lf", &temp[i*(*n)+j]);
			}
			MPI_Send(temp, local_m*(*n), MPI_DOUBLE, q, 0, MPI_COMM_WORLD);
		}

	}
	else
	{
		MPI_Recv(local_A, local_m*(*n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
	}

	return local_A;
	
}  /* Read_matrix */

void Parallel_matrix_matrix_prod(
		double local_A[],
		double local_B[],
		double local_C[], 
		int 	n,
		int 	local_n,
		int	local_m,
		int	l,
		int p,
		int my_rank)
		
{
	int count, i, j, k;
	MPI_Status status;

	for(count = 0; count < p; count++)
	{
		for (i=0; i<local_m; i++)
		{
			for(j=0; j<l; j++)			
			{
				if(count == 0)
					local_C[i*l+j] = 0;

				for(k=0; k<local_n; k++)
					local_C[i*l+j] = local_C[i*l+j] + local_A[i*n+k+((my_rank+count)%p)*local_n]*local_B[k*l+j];				
			}
		}
	
		int source = (my_rank+1)%p;
		int dest = (my_rank-1+p)%p;

		MPI_Sendrecv_replace(local_B, local_n*l, MPI_DOUBLE, dest, (my_rank+p)%p, source, (my_rank+1)%p, MPI_COMM_WORLD, &status); 

	}
}/* Parallel_matrix_matrix_prod */

void Print_matrix(
      char      title[]    /* in */,
      double    local_A[]  /* in */, 
      int       m          /* in */, 
      int       n          /* in */,
      int    	p          	/* in  */,
      int    	my_rank    	/* in  */) 
{
   int i, j, q;
   int local_m = m/p;
	MPI_Status status;
	FILE *ofp;
   

   if (my_rank == 0) 
	{
		ofp = fopen(title, "w"); 
		fprintf(ofp,"%d\n%d\n",m,n);
		for (i = 0; i<local_m; i++)
		{
			for (j=0;j<n; j++)
				fprintf(ofp,"%lf\n", local_A[i*n+j]);
		}

        	for (q = 1; q < p; q++) 
		{
			MPI_Recv(local_A, local_m*n, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, &status);
			for (i = 0; i<local_m; i++)
			{
				for (j=0;j<n; j++)
					fprintf(ofp,"%lf\n", local_A[i*n+j]);
			}
		}

	}
	else
	{
		MPI_Send(local_A, local_m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}  /* Print_matrix */
