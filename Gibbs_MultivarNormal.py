#Replication of a Gibbs sampling experiment from Hoff, Ch7

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

#Data Generation
n=22
mu=np.zeros((2,1))
mu[0]=47
mu[1]=54
Sigma=[[180,147],[147,240]]
Y=np.zeros((n,2))
for i in range(n):
    Y[i]=np.random.multivariate_normal([float(mu[0]),float(mu[1])], Sigma)
y=np

#Hyperparameter inputs
mu_0=np.zeros((2,1))
mu_0[0]=50
mu_0[1]=50
L_0=[[625,312.5],[312.5,625]]
nu_0=4
S_0=L_0

#From https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py
def sample_invwishart(S,nu):
    n=2
    chol=np.linalg.cholesky(S)
    x=np.random.randn(nu,n)
    R=np.linalg.qr(x,'r')
    T=scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return T @ T.T

#Gibbs Sampling Algorithm
num_iterations=5000
SSigma=np.zeros((num_iterations+1,2,2))
mu_n=np.zeros((num_iterations+1,2,1))
L_n=np.zeros((num_iterations+1,2,2))
theta=np.zeros((num_iterations+1,2,1))
S_n=np.zeros((num_iterations+1,2,2))
S_theta=np.zeros((num_iterations+1,2,2))
SSigma[0]=np.cov(np.transpose(Y))
sum_y=np.zeros((2,1))
for i in range(n):
    sum_y[0]+=Y[i][0]
    sum_y[1]+=Y[i][1]
for s in range(num_iterations):
    mu_n[s]=np.matmul(np.linalg.inv(np.linalg.inv(L_0)+n*np.linalg.inv(SSigma[s])),(np.matmul(np.linalg.inv(L_0),mu_0)+np.matmul(np.linalg.inv(SSigma[s]),sum_y)))
    L_n[s]=np.linalg.inv(np.linalg.inv(L_0)+n*np.linalg.inv(SSigma[s]))
    theta[s+1][0]=np.random.multivariate_normal([float(mu_n[s][0]),float(mu_n[s][1])], L_n[s])[0]
    theta[s+1][1]=np.random.multivariate_normal([float(mu_n[s][0]),float(mu_n[s][1])], L_n[s])[1]
    for i in range(n):
        S_theta[s][0][0]+=(float(Y[i][0]-theta[s+1][0]))**2
        S_theta[s][0][1]+=(float(Y[i][0]-theta[s+1][0]))*(float(Y[i][1]-theta[s+1][1]))
        S_theta[s][1][0]+=(float(Y[i][0]-theta[s+1][0]))*(float(Y[i][1]-theta[s+1][1]))
        S_theta[s][1][1]+=(float(Y[i][1]-theta[s+1][1]))**2
    S_n[s]=S_0+S_theta[s]
    SSigma[s+1]=sample_invwishart(S_n[s],nu_0+n)
theta_Gibbs=np.zeros((2,1))
Sigma_Gibbs=np.zeros((2,2))
for s in range(3000,5000):
    theta_Gibbs+=theta[s]
    Sigma_Gibbs+=SSigma[s]
theta_Gibbs/=2000
Sigma_Gibbs/=2000
print(theta_Gibbs)
print(Sigma_Gibbs)
