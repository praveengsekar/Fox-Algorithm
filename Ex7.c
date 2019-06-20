#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

typedef struct GRID_INFO_T
{
	int p;
	MPI_Comm comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
	int q;
	int my_row;
	int my_col;
	int my_rank;
}GRID_INFO_T;

void Setup_grid( GRID_INFO_T* grid)
{
	int dimensions[2];
	int wrap_around[2];
  	int coordinates[2];
  	int free_coords[2];

	/* set up global grid information */
	MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));

	/* We assume p is a perfect square */
	grid->q = (int) sqrt((double) grid->p);
	dimensions[0] = dimensions[1] = grid->q;

  	/* circular shift in second dimension, also in first just because */
  	wrap_around[0] = 1;
  	wrap_around[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->comm));
	MPI_Comm_rank(grid->comm, &(grid->my_rank));

	/* get process coordinates in grid communicator */
  	MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
	grid->my_row = coordinates[0];
	grid->my_col = coordinates[1];

	/* set up row communicator */
  	free_coords[0] = 0;
  	free_coords[1] = 1;
  	MPI_Cart_sub(grid->comm, free_coords, &(grid->row_comm));

	/* set up column communicator */
	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid->comm, free_coords, &(grid->col_comm));
}

double* Read_matrix(char* filename, GRID_INFO_T* grid, int* m, int* n)
{
	int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    double*     temp;
	int 		m_bar;
	int 		n_bar;
	double* 	local_A;
    MPI_Status status;
	FILE *ifp;

	if (grid->my_rank == 0)
	{
		ifp = fopen(filename, "r"); 
		fscanf(ifp,"%d", m);
		fscanf(ifp,"%d", n);		
	}

	MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
	m_bar = *m/grid->q;
	n_bar = *n/grid->q;

	local_A = malloc(m_bar*n_bar*sizeof(double));

    if (grid->my_rank == 0) 
	{
        temp = malloc(n_bar*sizeof(double));
        for (mat_row = 0;  mat_row < *m; mat_row++) 
		{
			grid_row = mat_row/(m_bar);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++)
			{
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0)
				{
                    for (mat_col = 0; mat_col < n_bar; mat_col++)
					{
						fscanf(ifp,"%lf", &local_A[mat_row*n_bar+mat_col]);
					}
                } 
				else{
                    for(mat_col = 0; mat_col < n_bar; mat_col++)
					{
                        fscanf(ifp,"%lf", temp + mat_col);
					}
                    MPI_Send(temp, n_bar, MPI_DOUBLE, dest, 0, grid->comm);
                }
            }
        }
        free(temp);
    } 
	else 
	{
        for (mat_row = 0; mat_row < m_bar; mat_row++) 
		{
			MPI_Recv(local_A+mat_row*n_bar, n_bar, MPI_DOUBLE, 0, 0, grid->comm, &status);		
		}
    }

	return local_A;
                     
}  /* Read_matrix */

void Print_matrix(char* title, double* local_A, GRID_INFO_T* grid, int m, int n) 
{
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    double*     temp;
    MPI_Status status;
	int m_bar = m/grid->q;
	int n_bar = n/grid->q;
	FILE *ofp;

    if (grid->my_rank == 0) 
	{
		ofp = fopen(title, "w"); 
		fprintf(ofp,"%d\n%d\n",m,n);
        temp = malloc(n_bar*sizeof(double));
        //printf("%s\n", title);
        for (mat_row = 0;  mat_row < m; mat_row++) 
		{
            grid_row = mat_row/m_bar;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) 
			{
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) 
				{
                    for(mat_col = 0; mat_col < n_bar; mat_col++)
                        fprintf(ofp,"%lf\n", local_A[mat_row*n_bar+mat_col]);
                } else 
				{
                    MPI_Recv(temp, n_bar, MPI_DOUBLE, source, 0, grid->comm, &status);
                    for(mat_col = 0; mat_col < n_bar; mat_col++)
                        fprintf(ofp,"%lf\n", *(temp + mat_col));
                }
            }
        }
        free(temp);
    } 
	else 
	{
        for (mat_row = 0; mat_row < m_bar; mat_row++)
			MPI_Send(local_A+mat_row*n_bar, n_bar, MPI_DOUBLE, 0, 0, grid->comm);
            //MPI_Send(&local_A[mat_row*n_bar], n_bar, MPI_DOUBLE, 0, 0, grid->comm);
    }
                     
} /* Print_matrix */

void Local_matrix_multiply( double*  local_A, double* local_B, double* local_C, int m_bar, int n_bar, int l_bar) 
{
	int i, j, k;

    for (i = 0; i < m_bar; i++)
        for (j = 0; j < l_bar; j++)
            for (k = 0; k < n_bar; k++)
				local_C[i*l_bar+j] = local_C[i*l_bar+j] + local_A[i*n_bar+k]*local_B[k*l_bar+j];                
}  /* Local_matrix_multiply */	

void Fox(int m_bar, int n_bar, int l_bar, GRID_INFO_T* grid, double* local_A, double* local_B, double* local_C)
{
	int stage;
	int kbar;
	int source;
	int dest;
	MPI_Status status;
	double* temp;
	
	int i, j;
	for (i = 0; i < m_bar; i++)
        for (j = 0; j < l_bar; j++)
            local_C[i*l_bar+j] = 0.0;

	dest = (grid->my_row - 1 + grid->q) % grid->q;
	source = (grid->my_row + 1) % grid->q;

	temp = malloc(m_bar*n_bar*sizeof(double));

	for(stage=0; stage < grid->q; stage++)
	{
		kbar = (grid->my_row + stage) % grid->q;
		if (kbar == grid->my_col)
		{
			MPI_Bcast(local_A, m_bar*n_bar, MPI_DOUBLE, kbar, grid->row_comm);
			Local_matrix_multiply(local_A, local_B, local_C,m_bar, n_bar, l_bar);
		}
		else
		{
			MPI_Bcast(temp, m_bar*n_bar, MPI_DOUBLE, kbar, grid->row_comm);
			Local_matrix_multiply(temp, local_B, local_C,m_bar, n_bar, l_bar);
		}
			
		MPI_Sendrecv_replace(local_B, n_bar*l_bar , MPI_DOUBLE, dest, 0, source, 0, grid->col_comm, &status);
	}

	free(temp);
} /* Fox */ 	


/*-------------------------------------------------------------------*/
main(int argc, char* argv[])
{
	int my_rank; /* rank of process */
   	int p; /* number of procescccccses */
	double* local_A;
	double* local_B;
	double* local_C;
	int m, n, l;
	int m_bar, n_bar, l_bar;
	double start, finish, istart, ifinish; 

	/* Start up MPI */
   	MPI_Init(&argc, &argv);
	
	struct GRID_INFO_T *grid = malloc(sizeof(struct GRID_INFO_T)); 
	
	Setup_grid(grid);

	MPI_Barrier(MPI_COMM_WORLD);
        if (grid->my_rank == 0)
                istart = MPI_Wtime();

	local_A = Read_matrix("A.txt", grid, &m, &n);
	local_B = Read_matrix("B.txt", grid, &n, &l);

	m_bar = m/grid->q;
	n_bar = n/grid->q;
	l_bar = l/grid->q;

	local_C = malloc(m_bar*l_bar*sizeof(double));

	MPI_Barrier(MPI_COMM_WORLD);
        if (grid->my_rank == 0)
                start = MPI_Wtime();
	Fox(m_bar, n_bar, l_bar, grid, local_A, local_B, local_C);
	MPI_Barrier(MPI_COMM_WORLD);
        if (grid->my_rank == 0)
                finish = MPI_Wtime();

	Print_matrix("C.txt", local_C, grid, m, l); 
	
	
	free(local_A);
	free(local_B);
	free(local_C);

	MPI_Barrier(MPI_COMM_WORLD);
        if (grid->my_rank == 0)
		{
			ifinish = MPI_Wtime();
			//printf("%lf\n",ifinish-istart);
			printf("%lf\n",finish-start);
		}
              
	free(grid);

	MPI_Finalize();

} /* main */