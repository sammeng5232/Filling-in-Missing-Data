#Replication of a Gibbs sampling experiment from Zhao et al., Bayesian CP Factorization of Incomplete Tensors with Automatic Rank Determination
#Version 1

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

data_array=np.loadtxt("current.csv", delimiter=",")

#Data Generation
#tensor is 3-mode
N1=122 #dimension of the first mode of the tensor
N2=46
N3=12
N=N1*N2*N3 #number of observations
NM=0 #number of missing observations
X=np.zeros((N1,N2,N3))
Y=np.zeros((N1,N2,N3))
B=np.zeros((N1,N2,N3))

for i in range(122):
    for j in range(46):
        for k in range(12):
            X[i][j][k]=data_array[j*12+k][i]
            B[i][j][k]=1
CSmean=X[118].mean()
CSstd=X[118].std()
for i in range(122):
    X[i]=((X[i]-X[i].mean())/X[i].std()+0)*1
for i in range(122):
    for j in range(46):
        for k in range(12):
            Y[i][j][k]=X[i][j][k]
print("The first entry of the CS series after standardization:",X[118][0][0])
for j in range(46):
    for k in range(12):
        if k%3!=1:
            B[118][j][k]=0
            Y[118][j][k]=0
            N-=1
            NM+=1

#Inititalization
R=10
a=2
b=2
tau=np.random.gamma(a,1/b)
c=2+np.zeros((R))
d=2+np.zeros((R))
lambdaa=np.zeros((R))
for r in range(R):
    lambdaa[r]=np.random.gamma(c[r],1/d[r])
U1=np.zeros((N1,R))
U2=np.zeros((N2,R))
U3=np.zeros((N3,R))
for j in range(N1):
    U1[j]=np.random.multivariate_normal(np.zeros((R)),np.diag(1/lambdaa))
for j in range(N2):
    U2[j]=np.random.multivariate_normal(np.zeros((R)),np.diag(1/lambdaa))
for j in range(N3):
    U3[j]=np.random.multivariate_normal(np.zeros((R)),np.diag(1/lambdaa))
                
#Gibbs sampling algorithm
num_iterations=60 #pick an even number
X_esti=np.zeros((N1,N2,N3))
for s in range(num_iterations):
    #updating lambda
    for r in range(R):
        ss=0
        for n1 in range(N1):
            ss+=U1[n1][r]**2
        for n2 in range(N2):
            ss+=U2[n2][r]**2
        for n3 in range(N3):
            ss+=U3[n3][r]**2
        lambdaa[r]=np.random.gamma(c[r]+(N1+N2+N3)/2,1/(d[r]+ss/2))
        
    #updating U
    #k=1
    for n1 in range(N1):
        Cov=np.zeros((R,R))
        for n2 in range(N2):
            for n3 in range(N3):
                if B[n1][n2][n3]==1:
                    Cov+=np.outer(np.multiply(U2[n2],U3[n3]),np.multiply(U2[n2],U3[n3]))
        Cov=np.linalg.inv(Cov*tau+np.outer(np.sqrt(lambdaa),np.sqrt(lambdaa)))
        mean=np.zeros((R))
        for n2 in range(N2):
            for n3 in range(N3):
                if B[n1][n2][n3]==1:
                    mean+=Y[n1][n2][n3]*np.multiply(U2[n2],U3[n3])
        mean=np.matmul(Cov,tau*mean)
        U1[n1]=np.random.multivariate_normal(mean,Cov)
    #k=2
    for n2 in range(N2):
        Cov=np.zeros((R,R))
        for n1 in range(N1):
            for n3 in range(N3):
                if B[n1][n2][n3]==1:
                    Cov+=np.outer(np.multiply(U1[n1],U3[n3]),np.multiply(U1[n1],U3[n3]))
        Cov=np.linalg.inv(Cov*tau+np.outer(np.sqrt(lambdaa),np.sqrt(lambdaa)))
        mean=np.zeros((R))
        for n1 in range(N1):
            for n3 in range(N3):
                if B[n1][n2][n3]==1:
                    mean+=Y[n1][n2][n3]*np.multiply(U1[n1],U3[n3])
        mean=np.matmul(Cov,tau*mean)
        U2[n2]=np.random.multivariate_normal(mean,Cov)
    #k=3
    for n3 in range(N3):
        Cov=np.zeros((R,R))
        for n1 in range(N1):
            for n2 in range(N2):
                if B[n1][n2][n3]==1:
                    Cov+=np.outer(np.multiply(U1[n1],U2[n2]),np.multiply(U1[n1],U2[n2]))
        Cov=np.linalg.inv(Cov*tau+np.outer(np.sqrt(lambdaa),np.sqrt(lambdaa)))
        mean=np.zeros((R))
        for n1 in range(N1):
            for n2 in range(N2):
                if B[n1][n2][n3]==1:
                    mean+=Y[n1][n2][n3]*np.multiply(U1[n1],U2[n2])
        mean=np.matmul(Cov,tau*mean)
        U3[n3]=np.random.multivariate_normal(mean,Cov)

    #updating tau
    X_curr=np.zeros((N1,N2,N3)) #recover the current tensor X from the current U
    for n1 in range(N1):
        for n2 in range(N2):
            for n3 in range(N3):
                for r in range(R):
                    X_curr[n1][n2][n3]+=U1[n1][r]*U2[n2][r]*U3[n3][r]
    ss=0
    for n1 in range(N1):
        for n2 in range(N2):
            for n3 in range(N3):
                if B[n1][n2][n3]==1:
                    ss+=(Y[n1][n2][n3]-X_curr[n1][n2][n3])**2
    tau=np.random.gamma(a+N/2,1/(b+ss/2))

    #saving the estimation in each iteration
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


























