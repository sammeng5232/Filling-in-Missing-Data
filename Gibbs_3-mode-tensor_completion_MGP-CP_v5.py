#Replication of a Gibbs sampling experiment from Rai et al. Scalable Bayesian Low-Rank Decomposition of Incomplete Multiway Tensors
#Version 5

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
N1=10 #dimension of the first mode of the tensor
N2=10
N3=10
N=N1*N2*N3 #number of observations
NM=0 #number of missing observations
M1=np.zeros((N))  #1st-mode-vectorized-coordinates of the missing entries
M2=np.zeros((N))  #2nd-mode-vectorized-coordinates of the missing entries
M3=np.zeros((N))  #3rd-mode-vectorized-coordinates of the missing entries
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
            X[i1][i2][i3]=((i1+2)*23+(i2+2)*29+(i3+2)*31)
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
                M1[NM]=int(i1*N2*N3+i2+i3*N2)
                M2[NM]=int(i2*N1*N3+i1+i3*N1)
                M3[NM]=int(i3*N1*N2+i1+i2*N1)
                NM+=1

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
            
#Vectorize Y
psi1=Y1.flatten()
psi2=Y2.flatten()
psi3=Y3.flatten()

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
#for r in range(R): #initialize U row-wise
    #U1[r]=np.random.multivariate_normal(mu1,Sigma1)
    #U2[r]=np.random.multivariate_normal(mu2,Sigma2)
    #U3[r]=np.random.multivariate_normal(mu3,Sigma3)
for j in range(N1): #initialize U column-wise
    V1[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
for j in range(N2):
    V2[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
for j in range(N3):
    V3[j]=np.random.multivariate_normal(np.zeros((R)),np.identity(R))
U1=np.transpose(V1)
U2=np.transpose(V2)
U3=np.transpose(V3)
tau_eps=np.random.gamma(a_0,1/b_0)
"""
X_init=np.zeros((N1,N2,N3)) #recover the initial tensor X from initial lambda and U
for i1 in range(N1):
    for i2 in range(N2):
        for i3 in range(N3):
            for r in range(R):
                X_init[i1][i2][i3]+=lambdaa[r]*U1[r][i1]*U2[r][i2]*U3[r][i3]"""

#Gibbs sampling algorithm
num_iterations=20
X_esti=np.zeros((N1,N2,N3))
for s in range(num_iterations):
    a=np.zeros((R,N1,N2,N3))
    b=np.zeros((R,N1,N2,N3))
    tau_head=np.zeros((R))
    mu_head=np.zeros((R))
    for r in range(R): #updating delta (and tau)
        ss=0
        for h in range(r, R):
            ss+=(lambdaa[h]**2)*np.prod(delta[0:h+1])/delta[r]
        delta[r]=np.random.gamma(a_c+(R-(r+1)-1)/2,1/(1+ss/2))
        tau[r]*=np.prod(delta[0:r+1])
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
    #updating U 
    #k=1
    V1=np.transpose(U1)
    W1=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V3,V2)))
    z1=V1.flatten()
    """ diag_mat is slow
    Co=W1
    for j in range(1,N1):
        Co=diag_mat([Co,W1])
    """
    Co=np.zeros((R*N1,N1*N2*N3))
    for j in range(0,N1):
        Co[R*j:R*(j+1),N2*N3*j:N2*N3*(j+1)]=W1
    for nm in range(NM):
        np.transpose(Co)[int(M1[nm])]=0
    Cov=np.linalg.inv(np.matmul(Co,np.transpose(Co))*tau_eps+np.identity(N1*R))
    mean=np.matmul(Cov,tau_eps*np.matmul(Co,psi1))    
    z1=np.random.multivariate_normal(mean,Cov)
    for j in range(N1):
        for r in range(R):
            V1[j][r]=z1[r+j*R]
    U1=np.transpose(V1)

    #k=2
    V2=np.transpose(U2)
    W2=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V3,V1)))
    z2=V2.flatten()
    Co=np.zeros((R*N2,N1*N2*N3))
    for j in range(0,N2):
        Co[R*j:R*(j+1),N1*N3*j:N1*N3*(j+1)]=W2
    for nm in range(NM):
        np.transpose(Co)[int(M2[nm])]=0
    Cov=np.linalg.inv(np.matmul(Co,np.transpose(Co))*tau_eps+np.identity(N2*R))
    mean=np.matmul(Cov,tau_eps*np.matmul(Co,psi2))
    z2=np.random.multivariate_normal(mean,Cov)
    for j in range(N2):
        for r in range(R):
            V2[j][r]=z2[r+j*R]
    U2=np.transpose(V2)

    #k=3
    V3=np.transpose(U3)
    W3=np.matmul(np.diag(lambdaa),np.transpose(scipy.linalg.khatri_rao(V2,V1)))
    z3=V3.flatten()
    Co=np.zeros((R*N3,N1*N2*N3))
    for j in range(0,N3):
        Co[R*j:R*(j+1),N1*N2*j:N1*N2*(j+1)]=W3
    for nm in range(NM):
        np.transpose(Co)[int(M3[nm])]=0
    Cov=np.linalg.inv(np.matmul(Co,np.transpose(Co))*tau_eps+np.identity(N3*R))
    mean=np.matmul(Cov,tau_eps*np.matmul(Co,psi3))
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
        X_esti+=X_curr
    print("Iteration",s+1,", the first entry of the current tensor:",X_curr[0][0][0])
X_esti/=(num_iterations/2)

#Results
Diff=X_esti-X
SqDiff=np.square(Diff)
MSE=SqDiff.sum()/(N1*N2*N3)
print("Mean squared error:")
print(MSE)


























