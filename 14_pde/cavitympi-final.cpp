#include <vector>
#include <iostream>
#include <fstream>
#include <mpi.h>

using namespace std;
typedef vector<vector<double>> matrix;

const int nx  = 41;
const int ny  = 41;
int lny     ;          // local array size
const int nt  = 500;
const int nit = 50;
const double dx  = (2.0 / (nx - 1));
const double dy  = (2.0 / (ny - 1));
const double dt  = 0.01;
const double rho = 1.0;
const double nu  = 0.02;


void exchange_halo(matrix &m, int rank, int size){
    MPI_Request request[2] ; 
    MPI_Status  status[2] ; 
    int request_count = 0 ; 
    if (rank > 0) {
        MPI_Isend(&m[1][0], nx, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[request_count]) ; 
        request_count++ ; 
    }
    if (rank < size -1) {
        MPI_Irecv(&m[lny-1][0], nx, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[request_count]) ; 
        request_count++ ; 
    }
  
    MPI_Waitall(request_count, request, status) ; 

    request_count = 0 ;
    if (rank < size - 1) {
        MPI_Isend(&m[lny-2][0], nx, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[request_count]) ; 
        request_count++ ; 
    }
    if (rank > 0) {
        MPI_Irecv(&m[0][0], nx, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[request_count]) ; 
        request_count++ ; 
    }

    MPI_Waitall(request_count, request, status) ; 
}


void sub1(matrix &b,matrix &v,matrix &u) {
    for (int j = 1; j < lny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            b[j][i] = rho * (1 / dt *\
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    (((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx))) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - (((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy))));
        }
    }
}

void sub2(matrix &pn,matrix &p,matrix &b,int rank, int size) {
    for (int it = 0; it < nit; it++) {
      for (int i = 0 ; i < lny; i++) {
            copy(p[i].begin(), p[i].end(), pn[i].begin());
        }
        for (int j = 1; j < lny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                           dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * dx*dx * dy*dy)
                          / (2 * (dx*dx + dy*dy));
            }
        }
        for (int i = 0; i < lny; i++) {
            p[i][nx-1] = p[i][nx-2];    //p[:, -1] = p[:, -2]
            p[i][0] = p[i][1];          //p[:, 0] = p[:, 1]
        }
        for (int i = 0; i < nx; i++) {
            p[0][i] = p[1][i];          //p[0, :] = p[1, :]
            p[lny-1][i] = 0;             //p[-1, :] = 0
        }

      // exchange halo line
      exchange_halo(p, rank, size) ; 
    }

    //printf("p=, %d, %f, %f, %f, %f\n",rank, p[0][10],p[1][10],p[lny-2][10],p[lny-1][10]) ; 
 
}

void sub3(matrix &u,matrix &un,matrix &vn,matrix &v,matrix &p, int rank, int size) {
    for (int i = 0; i < lny; i++) {
        copy(u[i].begin(), u[i].end(), un[i].begin());
        copy(v[i].begin(), v[i].end(), vn[i].begin());
    }
    for (int j = 1; j < lny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                               - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                               + nu * dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                               + nu * dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
            v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                               - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                               + nu * dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                               + nu * dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
        }
    }

    for (int i = 0; i < lny; i++) {
        u[i][nx-1]  = 0; 
        u[i][0]     = 0;
        v[i][nx-1]  = 0;    
        v[i][0]     = 0;
    }
    for (int i = 0; i < nx; i++) {
        u[lny-1][i]  = 1; 
        u[0][i]      = 0;  
        v[lny-1][i]  = 0;  
        v[0][i]      = 0;
    }
    exchange_halo(u, rank, size) ; 
    exchange_halo(v, rank, size) ; 

    //printf("u=, %d, %f, %f, %f, %f\n",rank, u[0][10],u[1][10], u[lny-2][10],u[lny-1][10]) ; 
    //printf("v=, %d, %f, %f, %f, %f\n",rank, v[0][10],v[1][10], v[lny-2][10],v[lny-1][10]) ; 
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank  ; 

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    lny = ny / size ; 

    if (rank == 0) {
      lny+=1 ; 
    } else if (rank == size -1) {
      lny= ny - lny * (size-1) + 1 ; 
    } else {
      lny+=2 ; 
    }
    
    if (size == 1) {
      lny = ny ; 
    }

    //printf("%d,%d,%d\n",rank,size,lny);

    matrix u(lny,vector<double>(nx)); 
    matrix v(lny,vector<double>(nx));
    matrix p(lny,vector<double>(nx));
    matrix b(lny,vector<double>(nx));
    matrix un(lny, vector<double>(nx));
    matrix vn(lny, vector<double>(nx));
    matrix pn(lny, vector<double>(nx));

    for (int j = 0; j < lny; j++) {
        for (int i = 0; i < nx; i++) {
            u[j][i] = 0.0;
            v[j][i] = 0.0;
            p[j][i] = 0.0;
            b[j][i] = 0.0;
        }
    }

    for (int i = 0; i < nt; i++) {
        sub1(b,v,u);
        sub2(pn,p,b,rank,size) ; 
        sub3(u,un,vn,v,p,rank,size) ; 
    }

    int js = 1 ; 
    if (rank == 0) js = 0 ;
    int je = lny - 1 ; 
    if (rank == size -1) je = lny ; 

    int j_offset = int(ny/size) * rank ; 
    if (rank == 0) j_offset = 0 ; 

    double send_buffer[ny][nx] ; 
    double p_all[ny][nx] ; 
    double u_all[ny][nx] ; 
    double v_all[ny][nx] ; 

    for (int j = 0 ; j < ny ; j++) {
        for (int i = 0 ; i < nx ; i++) {
            send_buffer[j][i] = 0.0 ; 
        }
    }

    for (int j = js; j < je; j++) {
        for (int i = 0; i < nx; i++) {
            send_buffer[j + -js + j_offset][i] = p[j][i] ; 
        }
    }
    MPI_Reduce(send_buffer, p_all, nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ; 

    for (int j = js; j < je; j++) {
        for (int i = 0; i < nx; i++) {
            send_buffer[j -js + j_offset][i] = u[j][i] ; 
        }
    }
    MPI_Reduce(send_buffer, u_all, nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ; 

    for (int j = js; j < je; j++) {
        for (int i = 0; i < nx; i++) {
            send_buffer[j + -js + j_offset][i] = v[j][i] ; 
        }
    }
    MPI_Reduce(send_buffer, v_all, nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ; 

    MPI_Finalize();

    if (rank != 0) exit(0) ; 

     
    ofstream fout ; 

    fout.open("p_mpi.bin", ios::out|ios::binary|ios::trunc);
    if (!fout) {
        cout << "file cannot open" ; 
        return 1 ; 
    }

    for (int j = 0 ; j < ny ; j++) {
        for (int i = 0 ; i < nx ; i++) {
            fout.write((char*)&p_all[j][i], sizeof(double)) ; 
        }
    }
    fout.close() ; 

    fout.open("u_mpi.bin", ios::out|ios::binary|ios::trunc);
    if (!fout) {
        cout << "file cannot open" ; 
        return 1 ; 
    }

    for (int j = 0 ; j < ny ; j++) {
        for (int i = 0 ; i < nx ; i++) {
            fout.write((char*)&u_all[j][i], sizeof(double)) ; 
        }
    }
    fout.close() ; 

    fout.open("v_mpi.bin", ios::out|ios::binary|ios::trunc);
    if (!fout) {
        cout << "file cannot open" ; 
        return 1 ; 
    }

    for (int j = 0 ; j < ny ; j++) {
        for (int i = 0 ; i < nx ; i++) {
            fout.write((char*)&v_all[j][i], sizeof(double)) ; 
        }
    }
    fout.close() ; 

    return 0;
} 
