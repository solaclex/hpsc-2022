#include <vector>
#include <iostream>
#include "mpi.h"

using namespace std;
typedef vector<vector<double>> matrix;

const int nx  = 41;
const int ny  = 41;
const int nt  = 500;
const int nit = 50;
const double dx  = (2.0 / (nx - 1));
const double dy  = (2.0 / (ny - 1));
const double dt  = 0.01;
const double rho = 1.0;
const double nu  = 0.02;


void sub1(matrix &b,matrix &v,matrix &u,int begin,int end) {
    for (int j = begin; j < end; j++) {
        for (int i = 1; i < nx-1; i++) {
            b[j][i] = rho * (1 / dt *\
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    (((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx))) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - (((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy))));
        }
    }
}

void sub2(matrix &pn,matrix &p,matrix &b,int begin,int end,int* stats) {
    for (int it = 0; it < nit; it++) {
        for (int i = begin-1; i < end+1; i++) {
            copy(p[i].begin(), p[i].end(), pn[i].begin());
        }
        for (int j = begin; j < end; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                           dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * dx*dx * dy*dy)
                          / (2 * (dx*dx + dy*dy));
            }
        }
        for (int i = stats[4]; i < stats[5]; i++) {
            p[i][nx-1] = p[i][nx-2];    //p[:, -1] = p[:, -2]
            p[i][0] = p[i][1];          //p[:, 0] = p[:, 1]
        }
        for (int i = 0; i < nx; i++) {
            p[0][i] = p[1][i];          //p[0, :] = p[1, :]
            p[ny-1][i] = 0;             //p[-1, :] = 0
        }
        if (stats[0] == 0 ) {
            MPI_Request request[2];
            MPI_Isend(&p[end-1][0], nx, MPI_DOUBLE, stats[0]+1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&p[end][0], nx, MPI_DOUBLE, stats[0]+1, 1, MPI_COMM_WORLD, &request[1]);
            MPI_Waitall(2, request, MPI_STATUS_IGNORE);
        } else if (stats[0] == stats[1]-1) {
            MPI_Request request[2];
            MPI_Isend(&p[begin][0], nx, MPI_DOUBLE, stats[0]-1, 1, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&p[begin-1][0], nx, MPI_DOUBLE, stats[0]-1, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Waitall(2, request, MPI_STATUS_IGNORE);
        } else {
            MPI_Request request[4];
            MPI_Isend(&p[begin][0], nx, MPI_DOUBLE, stats[3], 1, MPI_COMM_WORLD, &request[0]);
            MPI_Isend(&p[end-1][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[1]);
            MPI_Irecv(&p[begin-1][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(&p[end][0], nx, MPI_DOUBLE, stats[2], 1, MPI_COMM_WORLD, &request[3]);
            MPI_Waitall(4, request, MPI_STATUS_IGNORE);
        }
    }
}

void sub3(matrix &u,matrix &un,matrix &vn,matrix &v,matrix &p,int begin,int end,int* stats) {
    for (int i = begin-1; i < end+1; i++) {
        copy(u[i].begin(), u[i].end(), un[i].begin());
        copy(v[i].begin(), v[i].end(), vn[i].begin());
    }
    for (int j = begin; j < end; j++) {
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
    for (int i = stats[4]; i < stats[5]; i++) {
        u[i][nx-1]  = 0; 
        u[i][0]     = 0;
        v[i][nx-1]  = 0;    
        v[i][0]     = 0;
    }
    for (int i = 0; i < nx; i++) {
        u[ny-1][i]  = 1; 
        u[0][i]     = 0;  
        v[ny-1][i]  = 0;  
        v[0][i]     = 0;
    }
    MPI_Request request[8];
    if (stats[0] == 0) {
        MPI_Isend(&u[end-1][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&u[end][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[1]);
        MPI_Isend(&v[end-1][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&v[end][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[3]);
        MPI_Waitall(4, request, MPI_STATUS_IGNORE);
    } else if (stats[0] == stats[1] - 1) {
        MPI_Isend(&u[begin][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&u[begin-1][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[1]);
        MPI_Isend(&v[begin][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&v[begin-1][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[3]);
        MPI_Waitall(4, request, MPI_STATUS_IGNORE);
    } else {
        MPI_Isend(&u[end-1][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&u[end][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[1]);
        MPI_Isend(&v[end-1][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&v[end][0], nx, MPI_DOUBLE, stats[2], 0, MPI_COMM_WORLD, &request[3]);
        MPI_Isend(&u[begin][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[4]);
        MPI_Irecv(&u[begin-1][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[5]);
        MPI_Isend(&v[begin][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[6]);
        MPI_Irecv(&v[begin-1][0], nx, MPI_DOUBLE, stats[3], 0, MPI_COMM_WORLD, &request[7]);
        MPI_Waitall(8, request, MPI_STATUS_IGNORE);
    }

}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank ,begin ,end ,begin0 ,end0;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    begin = rank * ((ny-2) / size) + 1;
    end = (rank + 1) * ((ny-2) / size) + 1;

    begin0 = begin;
    end0 = end;

    if (rank == 0) {begin0 = begin-1;}
    if (rank == size -1) {end0 = end+1;}

    int recv_from = (rank + 1);
    int send_to = (rank - 1);

    int stats[] = {rank,size,recv_from,send_to,begin0,end0};

    matrix u(ny,vector<double>(nx)); 
    matrix v(ny,vector<double>(nx));
    matrix p(ny,vector<double>(nx));
    matrix b(ny,vector<double>(nx));
    matrix un(ny, vector<double>(nx));
    matrix vn(ny, vector<double>(nx));
    matrix pn(ny, vector<double>(nx));

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u[j][i] = 0.0;
            v[j][i] = 0.0;
            p[j][i] = 0.0;
            b[j][i] = 0.0;
        }
    }

    for (int i = 0; i < nt; i++) {
        sub1(b,v,u,begin,end);
        sub2(pn,p,b,begin,end,stats);
        sub3(u,un,vn,v,p,begin,end,stats);
    }
    MPI_Finalize();
     
    return 0;
} 