#Replication of a Gibbs sampling experiment from Chen et al.
#A Bayesian Tensor Decomposition Approach for Spatiotemporal Traffic Data Imputation，2019

"""
Process:
1. Create a random matrix X_full
2. Obtain data X from X_full with missing observations and errors
3. Initialize Hyperparameters
4. Update the CPD using Gibbs sampling
5. Obtain an estimate X_esti by computing the avg of X in the iterations after stabilizing
6. Compare X_esti with X_full by MSE
"""

import math
import nbi_stat
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import statistics
import scipy.stats as stats
import scipy.special as special
import scipy.linalg
from scipy.special import logsumexp

#From https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py
def sample_wishart(sigma, nu):
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (nu <= 81+n) and (nu == round(nu)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,nu)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(nu - np.arange(n))))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

#Data Generation
N=20
X_full=np.zeros((N,N,N,N))
X=np.zeros((N,N,N,N))
w=np.random.gamma(2,2)
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                X_full[i][j][k][l]=(i+2)*23+(j+2)*29+(k+2)*31+(l+2)*37
                X[i][j][k][l]=X_full[i][j][k][l]+np.random.normal(0,np.sqrt(1/w))
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                if (i+j+k+l)%5==1:
                    X[i][j][k][l]=0
B=np.zeros((N,N,N,N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                B[i][j][k][l]=1
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                if (i+j+k+l)%5==1:
                    B[i][j][k][l]=0

#Initialization
mu_0=np.zeros((N))
beta_0=1
W_0=np.identity(N)
R=N
nu_0=R
U=np.zeros((4,N,R))+np.random.normal(N,1) #mode=4
a_0=1
b_0=1
tau_eps=np.random.gamma(a_0,b_0)

#Gibbs sampling algorithm
num_iterations=10
X_esti=np.zeros((N,N,N,N))
for s in range(num_iterations):
    W_0_head=np.zeros((4,N,N))
    nu_0_head=np.zeros((4))
    L=np.zeros((4,N,N))
    mu_head=np.zeros((4,N))
    Mu_head=np.zeros((4,N,N))
    L_head=np.zeros((4,N,N))
    LL_head=np.zeros((4,N,N,N))
    mu=np.zeros((4,N))
    X_head=np.zeros((N,N,N,N))
    u_bar=np.zeros((4,N))
    S=np.zeros((4,N,N))
    for k in range(4):
        for n in range(N):
            u_bar[k]+=U[k][n]
        for n in range(N):
            S[k]+=np.outer(U[k][n]-u_bar[k],U[k][n]-u_bar[k])/N
        W_0_head[k]=np.linalg.inv(N*S[k]+(N*beta_0/(N+beta_0))*np.outer(u_bar[k]-mu_0,u_bar[k]-mu_0)+np.linalg.inv(W_0))
        nu_0_head[k]=N+nu_0
        L[k]=sample_wishart(W_0_head[k],math.floor(nu_0_head[k]))
        mu_head[k]=(N/(N+beta_0))*(u_bar[k]+beta_0*mu_0)
        L_head[k]=(N+beta_0)*L[k]
        mu[k]=np.random.multivariate_normal(mu_head[k],np.linalg.inv(L_head[k]))
        for i_k in range(N):
            LL_head[k][i_k]=np.zeros((N,N))
            Mu_head[k][i_k]=np.zeros(N)
            if k==0:
                for i_1 in range(N):
                    for i_2 in range(N):
                        for i_3 in range(N):
                            LL_head[k][i_k]+=B[i_k][i_1][i_2][i_3]*(np.outer(np.multiply(np.multiply(U[1][i_1],U[2][i_2]),U[3][i_3]),
                                                                             np.multiply(np.multiply(U[1][i_1],U[2][i_2]),U[3][i_3])))
                LL_head[k][i_k]=LL_head[k][i_k]*tau_eps+L[k]
            if k==1:
                for i_0 in range(N):
                    for i_2 in range(N):
                        for i_3 in range(N):
                            LL_head[k][i_k]+=B[i_0][i_k][i_2][i_3]*(np.outer(np.multiply(np.multiply(U[0][i_0],U[2][i_2]),U[3][i_3]),
                                                                             np.multiply(np.multiply(U[0][i_0],U[2][i_2]),U[3][i_3])))
                LL_head[k][i_k]=LL_head[k][i_k]*tau_eps+L[k]
            if k==2:
                for i_0 in range(N):
                    for i_1 in range(N):
                        for i_3 in range(N):
                            LL_head[k][i_k]+=B[i_0][i_1][i_k][i_3]*(np.outer(np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[3][i_3]),
                                                                             np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[3][i_3])))
                LL_head[k][i_k]=LL_head[k][i_k]*tau_eps+L[k]
            if k==3:
                for i_0 in range(N):
                    for i_1 in range(N):
                        for i_2 in range(N):
                            LL_head[k][i_k]+=B[i_0][i_1][i_2][i_k]*(np.outer(np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[2][i_2]),
                                                                             np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[2][i_2])))
                LL_head[k][i_k]=LL_head[k][i_k]*tau_eps+L[k]
            if k==0:
                for i_1 in range(N):
                    for i_2 in range(N):
                        for i_3 in range(N):
                            Mu_head[k][i_k]+=X[i_k][i_1][i_2][i_3]*np.multiply(np.multiply(U[1][i_1],U[2][i_2]),U[3][i_3])
                Mu_head[k][i_k]=np.matmul(np.linalg.inv(LL_head[k][i_k]),(Mu_head[k][i_k]*tau_eps+np.matmul(L[k],mu[k])))
            if k==1:
                for i_0 in range(N):
                    for i_2 in range(N):
                        for i_3 in range(N):
                            Mu_head[k][i_k]+=X[i_0][i_k][i_2][i_3]*np.multiply(np.multiply(U[0][i_0],U[2][i_2]),U[3][i_3])
                Mu_head[k][i_k]=np.matmul(np.linalg.inv(LL_head[k][i_k]),(Mu_head[k][i_k]*tau_eps+np.matmul(L[k],mu[k])))
            if k==2:
                for i_0 in range(N):
                    for i_1 in range(N):
                        for i_3 in range(N):
                            Mu_head[k][i_k]+=X[i_0][i_1][i_k][i_3]*np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[3][i_3])
                Mu_head[k][i_k]=np.matmul(np.linalg.inv(LL_head[k][i_k]),(Mu_head[k][i_k]*tau_eps+np.matmul(L[k],mu[k])))
            if k==3:
                for i_0 in range(N):
                    for i_1 in range(N):
                        for i_2 in range(N):
                            Mu_head[k][i_k]+=X[i_0][i_1][i_2][i_k]*np.multiply(np.multiply(U[0][i_0],U[1][i_1]),U[2][i_2])
                Mu_head[k][i_k]=np.matmul(np.linalg.inv(LL_head[k][i_k]),(Mu_head[k][i_k]*tau_eps+np.matmul(L[k],mu[k])))
            U[k][i_k]=np.random.multivariate_normal(Mu_head[k][i_k],np.linalg.inv(LL_head[k][i_k]))
    a_0_head=a_0
    b_0_head=b_0
    for i_0 in range(N):
        for i_1 in range(N):
            for i_2 in range(N):
                for i_3 in range(N):
                    for r in range(R):
                        X_head[i_0][i_1][i_2][i_3]+=U[0][i_0][r]*U[1][i_1][r]*U[2][i_2][r]*U[3][i_3][r]
    for i_0 in range(N):
        for i_1 in range(N):
            for i_2 in range(N):
                for i_3 in range(N):
                    if B[i_0][i_1][i_2][i_3]==1:
                        a_0_head+=0.5
                        b_0_head+=0.5*(X[i_0][i_1][i_2][i_3]-X_head[i_0][i_1][i_2][i_3])**2
    tau_eps=np.random.gamma(a_0_head,b_0_head)
    if s>=num_iterations/2:
        X_esti+=X_head/(num_iterations/2)

#Results
MSE=0
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                MSE+=(X_full[i][j][k][l]-X_esti[i][j][k][l])**2
MSE/=N**4
print("Mean squared error:")
print(MSE)










            
