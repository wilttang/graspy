import matplotlib.pyplot as plt
import numpy as np
from graspy.simulations import sbm
from numpy.random import normal, poisson
from scipy.stats import norm
import random
import itertools
from math import exp as ex
from math import log
import matplotlib.pyplot as plt
from load_dataset import load_COBRE
from graspy.plot import heatmap

def fit_cls(A,e,y,iter,a,rho,count):
    sampleNo,_,_ = A.shape
    R = np.ones((dim,N), dtype=float)/(dim*N)
    C = np.ones((N,1), dtype=float)/N
    b = 0
    ls = []

    for itr in range(1,iter):
        gradR = np.zeros((dim,N))/(dim*N)
        gradC = np.zeros((N,1))/N
        gradb = 0
        for k in range(0,sampleNo):
            tmp = b
            tmpGradR = np.zeros((dim,N)) 
            for i in range(0,dim):
                if y[k] == i:
                    tmp -= e[i,:].T@R@A[k,:,:]@C/count[i]
                    tmpGradR -= e[i,:].reshape(-1,1)@C.T@A[k,:,:].T/count[i]
                else:
                    tmp += e[i,:].T@R@A[k,:,:]@C/count[i]
                    tmpGradR += e[i,:].reshape(-1,1)@C.T@A[k,:,:].T/count[i]
            
            phi = ex(tmp)
            phi_ = phi/(1+phi)
            
            gradR += phi_*tmpGradR

        gradR = gradR/sampleNo + rho*np.sign(R)   
        R = R - a*gradR
        
        for k in range(0,sampleNo):
            tmp = b 
            tmpGradC = np.zeros((N,1))
            for i in range(0,dim):
                if y[k] == i:
                    tmp -= e[i,:].T@R@A[k,:,:]@C/count[i]
                    tmpGradC -= A[k,:,:].T@R.T@e[i,:].reshape(-1,1)/count[i]
                else:
                    tmp += e[i,:].T@R@A[k,:,:]@C/count[i]
                    tmpGradC += A[k,:,:].T@R.T@e[i,:].reshape(-1,1)/count[i]
            
            phi = ex(tmp)
            phi_ = phi/(1+phi)
            
            gradC += phi_*tmpGradC

        gradC = gradC/sampleNo + rho*np.sign(C)
        C = C - a*gradC
        
        for k in range(0,sampleNo):
            tmp = b 
            tmpGradC = np.zeros((N,1))
            for i in range(0,dim):
                if y[k] == i:
                    tmp -= e[i,:].T@R@A[k,:,:]@C/count[i]
                else:
                    tmp += e[i,:].T@R@A[k,:,:]@C/count[i]
            
            phi = ex(tmp)
            
            gradb += phi/(1+phi)
            
        b = b - a*gradb/sampleNo
        
    return R,C,b

#########################################################
def predictAccuracy(R,C,b,e,A,y):
    l = len(y)
    accuracy = 0

    for i in range(0,l):
        lbl = np.argmax(e@R@A[i,:,:]@C)
        if lbl == y[i]:
            accuracy += 1

    return accuracy*100/l

#########################################################
def getBase(dim):
    e = np.zeros((dim,dim),dtype=float)
    labelVec = np.asarray(range(1,dim+1),dtype=float)
    e[0,:] = labelVec/(labelVec@labelVec.T)**.5

    for i in range(1,dim):
        tmp = np.roll(labelVec,i)
        tmp = tmp.reshape(1,-1)
        tmp_ = tmp.copy()
        for j in range(0,i):
            eTmp = e[j,:].reshape(1,-1)
            tmp_ -= (tmp@eTmp.T/(eTmp@eTmp.T))*eTmp
            #print(tmp.shape)
        e[i,:] = tmp_/(tmp_@tmp_.T)**.5
    return e

##########################################################
crossValidate = 5
accuracy = 0

A, y = load_COBRE(ptr=True)
_,count = np.unique(y, return_counts=True)
dim = len(count)
e = getBase(dim)
totalSample, N, _ = A.shape

count = count/totalSample
indx = np.asarray(range(0,totalSample))
random.shuffle(indx)

testSample = int(totalSample/crossValidate)
for i in range(0,1):
    print("Doing test on %d fold.........\n"%(i+1))
    _,count = np.unique(y[0:totalSample-testSample], return_counts=True)
    count = count/(totalSample-testSample)

    print(count)
    indx = np.roll(indx,testSample)
    R,C,b = fit_cls(A[0:totalSample-testSample,:,:],e,y[0:totalSample-testSample],6000,1e-2,1e-6,count)
    accuracy += predictAccuracy(R,C,b,e,A[totalSample-testSample:totalSample,:,:],y[totalSample-testSample:totalSample])

    tm1 = R[1,:].reshape(-1,1)@C.T
    heatmap(tm1, title ='weight')

accuracy = accuracy/crossValidate
print("Total accuracy %f\n"%accuracy)





