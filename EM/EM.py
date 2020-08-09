import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heapq import nlargest
import math
import random

def GaussDistribution(x,mu,sigma, D):
    x = np.transpose(np.array([x]) ) # 2 X 1
    sigmaInverse =  np.linalg.inv(sigma)
    sigmaDet = math.fabs(np.linalg.det(sigma) )
    val = np.matmul(np.transpose((x - mu)),sigmaInverse )
    # print(val.shape)
    val = np.matmul(val,(x - mu))
    # print((x - mu).shape)
    val = np.multiply(val,-0.5)

    if val[0][0] > 500:
        print("Excess : ",val[0][0])
        val[0][0] = random.randint(0,500)
    if val[0][0] < -500:
        print("Too low : ",val[0][0])
        val[0][0] = -random.randint(0, 500)
        # val[0][0] = -500
    # val *= -0.5
    # print(val.shape)
    # print(sigmaDet)
    denom = math.sqrt(math.pow(2*math.pi,D) * sigmaDet)
    return float((1/denom) * math.exp(val[0][0]) )

def initializeRandomly(K,D):
    mu = []
    sigma = []
    w = []
    for k in range(K):
        mu.append(np.random.rand(D,1))
        isOk = False
        while True:
            mat = np.random.rand(D,D)
            if np.linalg.det(mat) > 0:
                break
        sigma.append(mat)
        w.append(float(1/K))
    return w,mu,sigma

def E_Step(x,mu,sigma,w,N,K,D):
    p = np.zeros((N,K) )

    for i in range(N):
        sum = 0
        for k in range(K):
            p[i][k] = w[k]*GaussDistribution(np.transpose(x[i] ),mu[k],sigma[k],D)
            sum += p[i][k]
        # for k in range(K):
        #     p[i][k] /= sum
        p[i] = np.multiply(p[i], float(1/sum) )
        # print(p)
    return p

def M_Step(x,p,N,K,D):
    mu = []
    sigma = []
    w = []
    for k in range(K):
        mu.append(np.zeros((D, 1)) )
        co_var_sum = np.zeros((D, D) )
        sum = 0
        for i in range(N):
            mu[k] = np.add(mu[k], p[i][k]*np.transpose(np.array([x[i] ]) ) )
            # mu[k] += p[i][k]*x[i]
            sum += p[i][k]
        # mu[k] /= sum
        mu[k] = np.multiply(mu[k], float(1/sum))
        w.append(float(sum/N) )
        for i in range(N):
            val = p[i][k] * np.matmul((np.transpose(np.array([x[i] ]))
                                       - mu[k]), np.transpose((np.transpose(np.array([x[i] ])) - mu[k])))
            co_var_sum = np.add(co_var_sum, val)
        co_var_sum  = np.multiply(co_var_sum, float(1/sum))
        # print("Vitore")
        # print(co_var_sum)
        sigma.append(co_var_sum)
    return w,mu,sigma

def evaluationValue(x,mu,sigma,w,N,K,D):
    loglikelihood = 0
    for i in range(N):
        val = 0
        for k in range(K):
            val += w[k] * GaussDistribution(np.transpose(x[i]), mu[k],sigma[k],D)
        loglikelihood += math.log(math.fabs(val) )
    return loglikelihood

def showClusteredData(df,K):
    data_array = np.array(df)
    dimension = len(df.columns)
    # print(dimension)
    data_mean = np.mean(data_array,axis=0,dtype=float) # make sure if it is okay
    # print(data_array)
    data_covar = np.cov(np.transpose(data_array) )
    # print(data_covar)

    eigenvalue, eigenvector = np.linalg.eig(data_covar)

    # print(eigenvalue)
    max_two = nlargest(K,eigenvalue)
    index_max_1 = list(eigenvalue).index(max_two[0])
    index_max_2 = list(eigenvalue).index(max_two[1])
    principal_components = np.empty([K,dimension])
    # print(principal_components.shape)
    row = [index_max_1,index_max_2]
    fPCA = []
    sPCA = []
    for e in eigenvector:
        fPCA.append(e[row[0] ] )
        sPCA.append(e[row[1] ] )

    principal_components = np.array([fPCA,sPCA]) # 2 X 100
    # 500 X 100 X 100 X 2 = 500 X 2
    projected_data_array = np.matmul(data_array,np.transpose(principal_components))
    x = projected_data_array[:,0]
    y = projected_data_array[:,1]

    plt.title("Clustered Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # plt.plot(x,y,'ro')
    # plt.axis([-20,20,-4,4])
    # plt.show()
    return projected_data_array

def EM(x,K,D):
    w,mu,sigma = initializeRandomly(K,D)
    N = len(x)
    prevLikelihood = 0
    _count = 0
    while(True):
        _count += 1
        print(_count)
        p = E_Step(x,mu,sigma,w,N,K,D)
        # print("E_STEP")
        w, mu, sigma = M_Step(x,p,N,K,D)
        # print("M_STEP")
        # print(sigma)
        newLikelihood = evaluationValue(x,mu,sigma,w,N,K,D)
        if math.fabs(newLikelihood - prevLikelihood) < 0.01 or _count > 100:
            print(newLikelihood, prevLikelihood, _count)
            break
        prevLikelihood = newLikelihood
    return p

def showClusteredDataFromEM(x,p,K):
    d1 = []
    d2 = []
    d3 = []
    for i in range(len(p)):
        _max = 0
        indx = 0
        for k in range(K):
            if p[i][k] > _max:
                _max = p[i][k]
                indx = k
        if indx == 0:
            d1.append(x[i])
        elif indx == 1:
            d2.append(x[i])
        elif indx == 2:
            d3.append(x[i])
    d1 = np.array(d1)
    d2 = np.array(d2)
    d3 = np.array(d3)
    print(d1.shape,d2.shape,d3.shape)
    d1_x = d1[:,0]
    d1_y = d1[:,1]
    plt.scatter(d1_x,d1_y,color='red')

    d2_x = d2[:, 0]
    d2_y = d2[:, 1]
    plt.scatter(d2_x, d2_y, color='blue')

    d3_x = d3[:, 0]
    d3_y = d3[:, 1]
    plt.scatter(d3_x, d3_y, color='green')

    plt.show()


def main():
    df = pd.read_csv('data.txt', delimiter='\t', header=None)
    np.random.seed(12345678)
    K = 3
    D = 2
    projected_data = showClusteredData(df,D)
    p = EM(projected_data,K,D)
    showClusteredDataFromEM(projected_data,p,K)

if __name__ == '__main__':
    main()



