#include <vector>
#include <iostream>

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

void sub1(matrix &b,matrix &v,matrix &u) {
#pragma omp parallel for collapse(2)
    for (int j = 1; j < ny-1; j++) {
        for (int i = 1; i < nx-1; i++) {
            b[j][i] = rho * (1 / dt *\
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    (((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx))) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - (((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy))));
        }
    }
}

void sub2(matrix &pn,matrix &p,matrix &b) {
    for (int it = 0; it < nit; it++) {
        for (int i = 0; i < ny; i++) {
            copy(p[i].begin(), p[i].end(), pn[i].begin());
        }
#pragma omp parallel for collapse(2)
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                           dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * dx*dx * dy*dy)\
                          / (2 * (dx*dx + dy*dy));
            }
        }
#pragma omp parallel for 
        for (int i = 0; i < ny; i++) {
            p[i][nx-1] = p[i][nx-2];    //p[:, -1] = p[:, -2]
            p[i][0] = p[i][1];          //p[:, 0] = p[:, 1]
        }
#pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            p[0][i] = p[1][i];          //p[0, :] = p[1, :]
            p[ny-1][i] = 0;             //p[-1, :] = 0
        }
    }
}

void sub3(matrix &u,matrix &un,matrix &vn,matrix &v,matrix &p) {
    for (int i = 0; i < ny; i++) {
        copy(u[i].begin(), u[i].end(), un[i].begin());
        copy(v[i].begin(), v[i].end(), vn[i].begin());
    }
#pragma omp parallel for collapse(2)
    for (int j = 1; j < ny-1; j++) {
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
#pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        u[i][nx-1]  = 0; 
        u[i][0]     = 0;
        v[i][nx-1]  = 0;    
        v[i][0]     = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        u[ny-1][i]  = 1; 
        u[0][i]     = 0;  
        v[ny-1][i]  = 0;  
        v[0][i]     = 0;
    }
}



int main() {
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
        sub1(b,v,u);
        sub2(pn,p,b);
        sub3(u,un,vn,v,p);
    }

    return 0;
} 