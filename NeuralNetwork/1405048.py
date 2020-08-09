import math
import random
from sklearn import preprocessing
import time
import datetime
import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self,N,L,K,w,v,y,Y,delta,mu):
        self.N = N # number of inputs
        self.L = L # number of layers
        self.K = K # list : number of neurons in each layer
        self.w = w # w[layer][neuron] = [each of the outputs of previous layer neuron]
        self.v = v # weighted sum of the weight and output for v[layer][input][neuron]
        self.y = y # output of each neuron K[r] X 1 matrix
        self.Y = Y # actual result
        self.J = 0 # initial cost estimate
        self.delta = delta
        self.mu = mu

    def f(self,x):
        choice = 1
        if choice == 1:
            return self.sigmoid(x)
        elif choice == 2:
            return self.relu(x)
        else:
            return self.tanh_f(x)

    def fDerivative(self,x):
        choice = 1
        if choice == 1:
            return self.sigmoid_derivative(x)
        elif choice == 2:
            return self.reluDerivative(x)
        else:
            return self.tanh_f_derivative(x)

    def sigmoid(self,x):
        a = 1
        return float(float(1) / (1 + math.exp(-a * x)) )

    def sigmoid_derivative(self,x):
        a = 1
        # return float(x)*(1-float(x) )
        return a * self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self,x):
        if x > 0:
            return float(x)
        else:
            return 0.0

    def reluDerivative(self,x):
        if x > 0:
            return 1.0
        else:
            return 0.0

    def tanh_f(self,x):
        return float(np.tanh(x) )

    def tanh_f_derivative(self,x):
        # return 1.0 - np.tanh(x) ** 2
        return float(1.0 - (np.tanh(x) * np.tanh(x)))

    def Initialize_Weight(self):
        for r in range(1,self.L+1):
            for j in range(self.K[r] + 1):
                self.w[r][j] = np.random.rand((self.K[r-1] + 1),1)
                self.w[r][j][0] = 0

    def ForwardComputation(self):
        err = []
        err.append(0) # 0 index is ignored
        for i in range(1, self.N + 1):
            for r in range(1, self.L + 1):
                for j in range(1, self.K[r] + 1):
                    self.v[r][i][j] = np.matmul(np.transpose(self.w[r][j]),self.y[r-1][i] )
                    self.y[r][i][j] = self.f(self.v[r][i][j])
            val = 0.0
            for m in range(1, self.K[self.L] + 1):
                # print(i,m)
                val += (self.f(self.v[self.L][i][m]) - self.Y[i][m])\
                       *(self.f(self.v[self.L][i][m]) - self.Y[i][m])
            # print("Val : ",val)
            err.append(0.5*val)
            self.J += err[i]
            ########## check if to stop training ? ######
            # don't think so

    def BackwardComputation(self):
        e = {} # check if to start from 1
        for r in range(self.L + 1):
            e[r] = {}
            for i in range(1, self.N + 1):
                e[r][i] = []
                if r == self.L:
                    for j in range(self.K[r] + 1):
                        e[r][i].append(0.0)
                else:
                    for j in range(self.K[r + 1] + 1):
                        e[r][i].append(0.0)

        for i in range(1, self.N + 1):
            for j in range(1, self.K[self.L] + 1):
                # print(j)
                e[self.L][i][j] = self.f(self.v[self.L][i][j]) - self.Y[self.L][j]
                self.delta[self.L][i][j] = e[self.L][i][j] \
                                           * self.fDerivative(self.v[self.L][i][j])
        # for i in range(1, self.N + 1):
            for r in reversed(range(2,(self.L + 1) )):
                for j in range(1, self.K[r] + 1):
                    for k in range(1, self.K[r] + 1):
                        # print(r,i,j)
                        e[r-1][i][j] += self.delta[r][i][k] * self.w[r][k][j]
                    self.delta[r-1][i][j] = e[r-1][i][j] \
                                            * self.fDerivative(self.v[r-1][i][j])

    def normalize(self,X,N):
        _max = -1
        _min = math.inf
        for x in X:
            if x >_max:
                _max = x
            if x <_min:
                _min = x
        if _max == _min:
            return X
        for i in range(N + 1):
            X[i] = (X[i] - _min) / (_max - _min)
            print(X[i])
        return X

    def Update_Weights(self):
        delW = {}
        for r in range(1,self.L + 1): # input layer NOT included
            delW[r] = {}
            for j in range(1, self.K[r] + 1):
                delW[r][j] = []
                for k in range(self.K[r-1] + 1):
                    delW[r][j].append(0.0)
        for r in range(1,self.L + 1):
            for j in range(1, self.K[r] + 1):
                for i in range (1, self.N + 1):
                    for k in range(self.K[r-1] + 1):
                        delW[r][j][k] += self.delta[r][i][j] * self.y[r-1][i][k]
                for k in range(self.K[r-1] + 1):
                    delW[r][j][k] *= -self.mu
                    self.w[r][j][k] += delW[r][j][k]
                # weight_df = pd.DataFrame(self.w[r])
                # col = weight_df.columns
                # self.w[r][j] = preprocessing.normalize(self.w[r][j])
                # self.w[r][j] = self.normalize(self.w[r][j], self.K[r-1])

        # print(self.w)
        #         for k in range(self.K[r-1] + 1):
        #             self.w[r][j][k] += delW[r][j][k]

    def Predicion(self,_count,y_test,Y_actual):
        match = 0
        not_match = 0
        v_temp = {}
        for r in range(1, self.L + 1):
            v_temp[r] = {}
            for i in range(1, _count + 1):
                v_temp[r][i] = []
                for j in range(self.K[r] + 1):
                    v_temp[r][i].append(0.0)
        for i in range(1, _count + 1):
            for r in range(1, self.L + 1):
                for j in range(1, self.K[r] + 1):
                    v_temp[r][i][j] = np.matmul(np.transpose(self.w[r][j]), y_test[r - 1][i])
                    y_test[r][i][j] = self.f(v_temp[r][i][j])
            _max = -1
            index = -1
            for j in range(1,self.K[self.L] + 1):
                if y_test[self.L][i][j] >= _max:
                    _max = y_test[self.L][i][j]
                    index = j
            if Y_actual[i-1] == index:
                match += 1
            else:
                not_match += 1
        print(match,not_match)
        accuracy = match/(match + not_match) * 100
        print("accuracy : ",accuracy)

    def TrainData(self):
        _count = 0
        _new_count = 0
        muCount = 0
        while True:
            _count += 1
            # print(_count)
            prevJ = self.J
            self.J = 0.0
            # print(1)
            self.ForwardComputation()
            if self.J < 100 or _count > 200 or (prevJ < self.J and _count > 1):
                print("Vitore J", _count, " : ", prevJ, self.J, self.mu)
                # self.mu /= 10
                # muCount += 1
                # if muCount > 3:
                #     break
                break
            # print(2)
            # if _count > _new_count + 50:
            print("J",_count," : ",prevJ, self.J)
                # _new_count = _count
            self.BackwardComputation()
            # print(3)
            self.Update_Weights()
            # for r in range(1,self.K[self.L]+1):
            #     for i in range(1,self.N + 1):
            #         print(self.y[r][i][0])

            # print(4)



def initializeAll():
    # df = pd.read_csv('trainNN.txt', delimiter='\t', header=None)
    df = pd.read_csv('Data/trainNN.txt', delimiter='\t', header=None)
    df_array = np.array(df)

    x = df_array[:, :-1]
    y = df_array[:, -1]

    # x = preprocessing.scale(x)

    x_df = pd.DataFrame(x)
    names = x_df.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(x_df)
    x_stan_df = pd.DataFrame(scaled_df, columns=names)
    x = x_stan_df.values

    featureCount = len(df.columns) - 1
    classCount = len(list(set(y)))
    N = df.shape[0]
    # print(N)

    Y = {}
    for i in range(1, N + 1):
        Y[i] = []
        Y[i].append(0.0)
        for j in range(1, classCount + 1):
            Y[i].append(0.0)
        Y[i][int(y[i - 1])] = 1.0 # otherwise needs label encoding


    K = []
    K.append(featureCount)  # K[0]
    # 1st neuronCount is number of features
    # last one is number of classes
    L = 3
    neuronCount = [featureCount, 3, classCount]  # SO THIS MEANS NUMBER OF CLASSES IS 5
    for i in neuronCount:
        K.append(i)  # at each Layer starting from 1 : number of neurons
    w = {}
    for r in range(1, L + 1):  # input layer NOT included but not needed i guess
        w[r] = {}
        for j in range(1, K[r] + 1):
            w[r][j] = np.random.rand((K[r - 1] + 1), 1)
            # w[r][j] = np.zeros(( (K[r - 1] + 1), 1 ) )
            # w[r][j][0] = 0  # threshold
            # w[r][j] = []
            # for k in range(K[r - 1] + 1):
            #     w[r][j].append(0.0)
    v = {}
    for r in range(1, L + 1):
        v[r] = {}
        for i in range(1, N + 1):
            v[r][i] = []
            for j in range(K[r] + 1):
                v[r][i].append(0.0)
    y = {}
    for r in range(L + 1):
        y[r] = {}
        for i in range(1, N + 1):
            y[r][i] = np.zeros((K[r] + 1, 1))
            y[r][i][0] = 1.0
    for i in range(1, N + 1):
        for k in range(featureCount):
            y[0][i][k + 1] = x[i-1][k]

    delta = {}  # check if to start from 1
    for r in range(1, L+1):
        delta[r] = {}
        for i in range(1, N + 1):
            delta[r][i] = []
            if r == L:
                for j in range(K[r] + 1):
                    delta[r][i].append(0.0)
            else:
                for j in range( K[r+1] + 1):
                    delta[r][i].append(0.0)
    mu = 0.001
    return N,L,K,w,v,y,Y,delta,mu

def initializeForPrediction(L,K):
    # df = pd.read_csv('testNN.txt', delimiter='\t', header=None)
    df = pd.read_csv('Data/testNN.txt', delimiter='\t', header=None)
    df_array = np.array(df)

    x = df_array[:, :-1]
    y = df_array[:, -1]

    # x = preprocessing.scale(x)

    x_df = pd.DataFrame(x)
    names = x_df.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(x_df)
    x_stan_df = pd.DataFrame(scaled_df, columns=names)
    x = x_stan_df.values

    featureCount = len(df.columns) - 1
    # classCount = len(list(set(y)))
    N = df.shape[0]
    # print(N)

    y_test = {}
    for r in range(L + 1):
        y_test[r] = {}
        for i in range(1, N + 1):
            y_test[r][i] = np.zeros((K[r] + 1, 1))
            y_test[r][i][0] = 1.0
    for i in range(1, N + 1):
        for k in range(featureCount):
            y_test[0][i][k + 1] = x[i - 1][k]

    return N,y_test,y

def main():
    FMT = '%H:%M:%S'

    N, L, K, w, v, y, Y, delta, mu = initializeAll()
    neuralNetwork = NeuralNetwork(N,L,K,w,v,y,Y,delta,mu)
    time_start = time.time()
    neuralNetwork.TrainData()
    time_end = time.time()
    # tdelta = datetime.strptime(time_end, FMT) - datetime.strptime(time_start, FMT)
    training_time = time_end - time_start
    print("Training Time : ",training_time)
    _count,y_test,Y_actual = initializeForPrediction(L,K)
    neuralNetwork.Predicion(_count,y_test,Y_actual)

if __name__ == '__main__':
    main()
