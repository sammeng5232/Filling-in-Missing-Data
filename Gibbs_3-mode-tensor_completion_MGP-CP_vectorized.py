#Replication of a Gibbs sampling experiment from Rai et al. Scalable Bayesian Low-Rank Decomposition of Incomplete Multiway Tensors

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

#from https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array
def diag_mat(rem=[], result=np.empty((0, 0))):
    if not rem:
        return result
    m = rem.pop(0)
    result = np.block(
        [
            [result, np.zeros((result.shape[0], m.shape[1]))],
            [np.zeros((m.shape[0], result.shape[1])), m],
        ]
    )
    return diag_mat(rem, result)

#Data Generation
#tensor is 3-mode
N1=20 #dimension of the first mode of the tensor
N2=20
N3=20
N=N1*N2*N3 #number of observations
mu1=np.zeros((N1))
mu2=np.zeros((N2))
mu3=np.zeros((N3))
Sigma1=np.identity(N1)
Sigma2=np.identity(N2)
Sigma3=np.identity(N3)
X=np.zeros((N1,N2,N3))
Y=np.zeros((N1,N2,N3))
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            X[i1][i2][i3]=(i1+2)*23+(i2+2)*29+(i3+2)*31
            Y[i1][i2][i3]=X[i1][i2][i3]+np.random.normal(0,np.sqrt(1/np.random.gamma(2,1/2))) #paper: rate parameter; here: scale parameter
B=np.zeros((N1,N2,N3))
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            B[i1][i2][i3]=1
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            if (i1+i2+i3)%4==1: # about 25% of the data are missing
                B[i1][i2][i3]=0
                N-=1
# if 50% of the data are missing, even 5000 iterations won't save it
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            if B[i1][i2][i3]==0: # about 25% of the data are missing
                Y[i1][i2][i3]=0

#Matricize Y
Y1=np.zeros((N1,N2*N3))
Y2=np.zeros((N2,N1*N3))
Y3=np.zeros((N3,N1*N2))
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            Y1[i1][i2+i3*N2]=Y[i1][i2][i3] #starting from 0
            Y2[i2][i1+i3*N1]=Y[i1][i2][i3]
            Y3[i3][i1+i2*N1]=Y[i1][i2][i3]

#Inititalization
a_0=1
b_0=1
a_c=2
R=5
delta=np.zeros((R))
tau=np.zeros((R))+1
lambdaa=np.zeros((R))
for r in range(R):
    delta[r]=np.random.gamma(a_c,1)
for r in range(R):
    for l in range(r+1):
        tau[r]*=delta[l]
for r in range(R):
    lambdaa[r]=np.random.normal(0,np.sqrt(1/tau[r]))
U1=np.zeros((R,N1))
U2=np.zeros((R,N2))
U3=np.zeros((R,N3))
V1=np.zeros((N1,R))
V2=np.zeros((N2,R))
V3=np.zeros((N3,R))
z1=np.zeros((N1*R))
z2=np.zeros((N2*R))
z3=np.zeros((N3*R))
#for r in range(R):
    #U1[r]=np.random.multivariate_normal(mu1,Sigma1)
    #U2[r]=np.random.multivariate_normal(mu2,Sigma2)
    #U3[r]=np.random.multivariate_normal(mu3,Sigma3)
for j in range(N1):
    V1[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
for j in range(N2):
    V2[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
for j in range(N3):
    V3[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
U1=np.transpose(V1)
U2=np.transpose(V2)
U3=np.transpose(V3)
tau_eps=np.random.gamma(a_0,1/b_0)
X_init=np.zeros((N1,N2,N3)) #recover the initial tensor X from initial lambda and U
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            for r in range(R):
                X_init[i1][i2][i3]+=lambdaa[r]*U1[r][i1]*U2[r][i2]*U3[r][i3]

#Gibbs sampling algorithm
num_iterations=40
X_esti=np.zeros((N1,N2,N3))
for s in range(num_iterations):
    a=np.zeros((R,N1,N2,N3))
    b=np.zeros((R,N1,N2,N3))
    tau=np.zeros((R))+1
    tau_head=np.zeros((R))
    mu_head=np.zeros((R))
    c1=np.zeros((R,N1,N2,N3))
    d1=np.zeros((R,N1,N2,N3))
    ttau1=np.zeros((R,N1))
    T1=np.zeros((R,N1,N1))
    SSigma1_head=np.zeros((R,N1,N1))
    alpha1=np.zeros((R,N1))
    Alpha1=np.zeros((R,N1,1)) #Alpha1[r] is a column vector
    Mu1_head=np.zeros((R,N1))
    c2=np.zeros((R,N1,N2,N3))
    d2=np.zeros((R,N1,N2,N3))
    ttau2=np.zeros((R,N2))
    T2=np.zeros((R,N2,N2))
    SSigma2_head=np.zeros((R,N2,N2))
    alpha2=np.zeros((R,N2))
    Alpha2=np.zeros((R,N2,1)) #Alpha2[r] is a column vector
    Mu2_head=np.zeros((R,N2))
    c3=np.zeros((R,N1,N2,N3))
    d3=np.zeros((R,N1,N2,N3))
    ttau3=np.zeros((R,N3))
    T3=np.zeros((R,N3,N3))
    SSigma3_head=np.zeros((R,N3,N3))
    alpha3=np.zeros((R,N3))
    Alpha3=np.zeros((R,N3,1)) #Alpha3[r] is a column vector
    Mu3_head=np.zeros((R,N3)) 
    for r in range(R): #updating delta
        ss=0
        for h in range(r, R):
            pp=1
            for l in range(0, r):
                pp*=delta[l]
            for l in range(r+1, h+1):
                pp*=delta[l]
            ss+=(lambdaa[h]**2)*pp
        delta[r]=np.random.gamma(a_c+(R-(r+1)-1)/2,1/(1+ss/2))
        for l in range(r+1):
            tau[r]*=delta[l]
    for r in range(R): #updating lambda
        for i1 in range(N1):
            for i2 in range(N2):
                for i3 in range(N3):
                    if B[i1][i2][i3]==1:
                        a[r][i1][i2][i3]=U1[r][i1]*U2[r][i2]*U3[r][i3]
                        for r_dash in range(R):
                            b[r][i1][i2][i3]+=lambdaa[r_dash]*U1[r_dash][i1]*U2[r_dash][i2]*U3[r_dash][i3]
                        b[r][i1][i2][i3]-=lambdaa[r]*U1[r][i1]*U2[r][i2]*U3[r][i3]
        for i1 in range(N1):
            for i2 in range(N2):
                for i3 in range(N3):
                    if B[i1][i2][i3]==1:
                        tau_head[r]+=a[r][i1][i2][i3]**2
                        mu_head[r]+=a[r][i1][i2][i3]*(Y[i1][i2][i3]-b[r][i1][i2][i3])
        tau_head[r]=(tau_head[r]*tau_eps)+tau[r]
        mu_head[r]*=tau_eps/tau_head[r]
        lambdaa[r]=np.random.normal(mu_head[r],np.sqrt(1/tau_head[r]))
    #updating U (and V=U^T)
    #k=1
    V1=np.transpose(U1)
    W1=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V3,V2)))
    z1=V1.flatten()
    e=np.zeros((N1,R,R))
    f=np.zeros((N1,R))
    for j in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                if B[j][i2][i3]==1:
                    e[j]+=np.outer(np.transpose(W1)[i2+i3*N2],np.transpose(W1)[i2+i3*N2])
                    f[j]+=Y1[j][i2+i3*N2]*np.transpose(W1)[i2+i3*N2]
        if j==0:
            Cov=e[j]
            mean=f[j]
        else:
            Cov=diag_mat([Cov,e[j]])
            mean=np.concatenate([mean,f[j]])
    Cov=np.linalg.inv(Cov*tau_eps+np.identity(N1*R))
    mean=np.matmul(Cov,mean*tau_eps)
    z1=np.random.multivariate_normal(mean,Cov)
    for j in range(N1):
        for r in range(R):
            V1[j][r]=z1[r+j*R]
    U1=np.transpose(V1)

    #k=2
    V2=np.transpose(U2)
    W2=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V3,V1)))
    z2=V2.flatten()
    e=np.zeros((N2,R,R))
    f=np.zeros((N2,R))
    for j in range(N2):
        for i1 in range(N1):
            for i3 in range(N3):
                if B[i1][j][i3]==1:
                    e[j]+=np.outer(np.transpose(W2)[i1+i3*N1],np.transpose(W2)[i1+i3*N1])
                    f[j]+=Y2[j][i1+i3*N1]*np.transpose(W2)[i1+i3*N1]
        if j==0:
            Cov=e[j]
            mean=f[j]
        else:
            Cov=diag_mat([Cov,e[j]])
            mean=np.concatenate([mean,f[j]])
    Cov=np.linalg.inv(Cov*tau_eps+np.identity(N2*R))
    mean=np.matmul(Cov,mean*tau_eps)
    z2=np.random.multivariate_normal(mean,Cov)
    for j in range(N2):
        for r in range(R):
            V2[j][r]=z2[r+j*R]
    U2=np.transpose(V2)

    #k=3
    V3=np.transpose(U3)
    W3=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V2,V1)))
    z3=V3.flatten()
    e=np.zeros((N3,R,R))
    f=np.zeros((N3,R))
    for j in range(N3):
        for i1 in range(N1):
            for i2 in range(N2):
                if B[i1][i2][j]==1:
                    e[j]+=np.outer(np.transpose(W3)[i1+i2*N1],np.transpose(W3)[i1+i2*N1])
                    f[j]+=Y3[j][i1+i2*N1]*np.transpose(W3)[i1+i2*N1]
        if j==0:
            Cov=e[j]
            mean=f[j]
        else:
            Cov=diag_mat([Cov,e[j]])
            mean=np.concatenate([mean,f[j]])
    Cov=np.linalg.inv(Cov*tau_eps+np.identity(N3*R))
    mean=np.matmul(Cov,mean*tau_eps)
    z3=np.random.multivariate_normal(mean,Cov)
    for j in range(N3):
        for r in range(R):
            V3[j][r]=z3[r+j*R]
    U3=np.transpose(V3)
        
    X_curr=np.zeros((N1,N2,N3)) #recover the current tensor X from current lambda and U
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for r in range(R):
                    X_curr[i1][i2][i3]+=lambdaa[r]*U1[r][i1]*U2[r][i2]*U3[r][i3]
    tt=0
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                if B[i1][i2][i3]==1:
                    tt+=(X_curr[i1][i2][i3]-Y[i1][i2][i3])**2
    tau_eps=np.random.gamma(a_0+N/2,1/(b_0+tt/2)) #updating tau_eps
    if s>=num_iterations/2:
        for i1 in range(N1):
            for i2 in range(N2):
                for i3 in range(N3):
                    X_esti[i1][i2][i3]+=X_curr[i1][i2][i3]/(num_iterations/2)
    print("Iteration",s+1,", the first entry of the current tensor:",X_curr[0][0][0])

#Results
MSE=0
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            MSE+=(X_esti[i1][i2][i3]-X[i1][i2][i3])**2/(N1*N2*N3)
print("Mean squared error:")
print(MSE)


























