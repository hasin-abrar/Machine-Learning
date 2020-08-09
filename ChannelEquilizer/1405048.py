import math
import numpy as np
import random
import pickle
class States:
    def __init__(self,n,l):
        self.n = n
        self.l = l
        self.states = [] # 1,2,3, ...
        self.value = {} # 000,0001, ...
        self.cluster_class = {} # 0 or 1
        self.transition = {} # binary prev_states are stored
        self.clusters = {}
        self.cluster_mean = {}
        self.cluster_variance = {}
        self.transition_cost = {}
        self.probability = {}

    def initializeStates(self):
        n = self.n
        l = self.l
        # self.states.append(0)
        for i in range(int(math.pow(2,n+l-1) ) ):
            self.states.append(i+1)
            binString = "{0:b}".format(i).zfill(n + l -1)
            self.value[i+1] = binString
            self.clusters[binString] = []
            self.transition[binString] = [] # prev two state will be 0.5
            if binString[0] == "0":
                self.cluster_class[binString] = "0"
            else:
                self.cluster_class[binString] = "1"


    def transitionProb(self):
        n = self.n
        l = self.l
        _count = -1
        for i in range(int(math.pow(2,n + l -1) ) ):
            state = self.states[i]
            binState = self.value[state]
            # if i % 2 == 0:
            #     nextStates = [ self.states[int(i / 2)], self.states[i // 2 + int(math.pow(2,n+l-2)) ] ]
            # else:
            #     nextStates = [ self.states[int((i -1) / 2) ], self.states[(i - 1) // 2 + int(math.pow(2,n+l-2)) ] ]
            if i == int(math.pow(2,n + l - 2)):
                _count = -1

            prevStates = [ self.states[_count + 1], self.states[_count + 2] ] # number of the states, not object
            _count += 2
            # self.transition[binState].append(nextStates)
            self.transition_cost[binState] = {}
            prevStates1 = self.value[prevStates[0]]
            prevStates2 = self.value[prevStates[1]]
            prevStates = [prevStates1,prevStates2]
            # print(prevStates)
            self.transition[binState].append(prevStates)
            self.transition_cost[binState][prevStates1] = 0.5 # P (binState | prevstate1) = 0.5
            self.transition_cost[binState][prevStates2] = 0.5

class Channel:

    def __init__(self, h, mean, variance):
        self.h = h
        self.mean = mean
        self.variance = variance


    # Ik needs to be n + L - 1 unit long
    def Simulate(self, Ik, L):
        xk = []
        ik = 0
        for i in range(L):
            val = 0.0
            for j in range(len(self.h)):
                val += self.h[j] * float(Ik[ik + j] )
            val += np.random.normal(self.mean, math.sqrt(self.variance))
            xk.append(val)
            ik += 1
        return xk

class Transmission:

    def __init__(self, filePath):
        self.x = []
        self.filePath = filePath

    def startTransmission(self, n, L, channel, _states):
        with open(self.filePath) as fp:
            line = fp.readline()
        for i in range((n - L)):
            line += "0"
        for i in reversed(range((n + L - 1), len(line))):
            Ik = []
            for j in range((n + L - 1)):
                Ik.append(line[i - j])
            Ik = list(reversed(Ik) )
            xk = channel.Simulate(Ik, L)
            state = ''.join(Ik)
            cluster = _states.clusters[state]
            cluster.append(xk)
            self.x.append(xk)
        # xn, ..., x1
        self.x = list(reversed(self.x) ) # may be not needed [if speed issue]
        for state in _states.states:
            stateString = _states.value[state]
            cluster = np.array(_states.clusters[stateString])

            # if state == 1:
            #     print(cluster, np.shape(cluster))

            _states.cluster_mean[stateString] = np.mean(cluster,axis=0)
            _states.cluster_variance[stateString] = np.cov(np.transpose(cluster) )
            _states.probability[stateString] = float(len(cluster)/ len(line) )

# 2
# 1 2 3
# 3 4

class Equilizer:
    def __init__(self, filepath,states,isDistanceUsed):
        self.x = []
        self.filePath = filepath
        self.path = {}
        self.cost = {}
        self.pathClass = {}
        self.isDistanceUsed = isDistanceUsed
        self.states = states

    def checkAccuracy(self, testFilePath, outputPath):
        with open(testFilePath) as fp:
            testLine = fp.readline()
        with open(outputPath) as fp:
            outputLine = fp.readline()
        _count = 0
        for i in range(len(testLine)):
            if testLine[i] == outputLine[i]:
                _count += 1
        accuracy = float(_count/len(testLine)) * 100
        print("Accuracy : ",accuracy,"%")

    def startEquilizing(self, n, L, channel):
        isDistanceUsed = self.isDistanceUsed
        with open(self.filePath) as fp:
            line = fp.readline()
        for i in range((n + L - 2)):
            line += "0"
        _count = 0
        for i in reversed(range((n + L - 2), len(line))):
            Ik = []
            _count += 1
            for j in range((n + L - 1)):
                Ik.append(line[i - j])
            Ik = list(reversed(Ik) )
            xk = channel.Simulate(Ik, L)
            if i == len(line)-1:
                # print("ami")
                if isDistanceUsed:
                    self.Viterbi_Distance(xk,True)
                else:
                    self.Viterbi(xk,True)
            else:
                if isDistanceUsed:
                    self.Viterbi_Distance(xk,False)
                else:
                    self.Viterbi(xk, False)
        print(len(line),"COUNT : ",_count)


        if isDistanceUsed:
            outputPath = "test_Distance_output.txt"
            min_cost = math.inf
            cluster = 1
            for state in self.states.states:
                binState = self.states.value[state]
                _cost = self.cost[binState]
                print(binState, _cost)
                if _cost < min_cost:
                    min_cost = _cost
                    cluster = binState
            print(min_cost, cluster, len(self.pathClass[cluster]))
        else:
            outputPath = "test_output.txt"
            max_cost = -math.inf
            cluster = 1
            for state in self.states.states:
                binState = self.states.value[state]
                _cost = self.cost[binState]
                print(binState, _cost)
                if _cost > max_cost:
                    max_cost = _cost
                    cluster = binState
            print(max_cost, cluster, len(self.pathClass[cluster]))

        with open(outputPath, "w") as fp:
            for c in reversed(range(len(self.pathClass[cluster]))):
                fp.write(self.pathClass[cluster][c])
                # fp.write(str(c ) )
        print(outputPath)
        self.checkAccuracy(self.filePath,outputPath)

    def GaussProbability(self,x, mu, sigma, L = 2):

        # x = np.transpose(np.array([x]))  # 2 X 1
        sigmaInverse = np.linalg.inv(sigma)
        sigmaDet = math.fabs(np.linalg.det(sigma))
        if sigmaDet == 0:
            print("Sigma Det 0")
            sigmaDet = 0.00000001
        val = np.matmul(np.transpose((np.subtract(x ,mu ))), sigmaInverse)
        val = np.matmul(val, np.subtract(x , mu))
        val = np.multiply(val, -0.5)

        if val > 500:
            # print("Excess : ", val[0][0])
            val = 500
        if val < -500:
            # print("Too low : ", val[0][0])
            val = -500
        denom = math.sqrt(math.pow(2 * math.pi, L) * sigmaDet)
        return float((1.0 / denom) * math.exp(val))

    def Viterbi(self, xk, isFirst):
        temp_cost = {}
        temp_main_path = {}
        temp_main_pathClass = {}
        if not isFirst:
            for state in self.states.states:
                binState = self.states.value[state]
                mean = self.states.cluster_mean[binState]
                sigma = self.states.cluster_variance[binState]
                # print("baire", np.shape(sigma))
                # can there be a domain error
                _cost = math.log(self.GaussProbability(xk, mean, sigma) )
                prevStates = self.states.transition[binState]

                # math.log(self.states.transition_cost[binState][prevStates[0][0]])
                _cost_1 = + math.log(0.5) + \
                          self.cost[prevStates[0][0] ] + _cost
                # math.log(self.states.transition_cost[binState][prevStates[0][1]])
                _cost_2 = math.log(0.5) + \
                          self.cost[prevStates[0][1]]  + _cost

                if _cost_1 > _cost_2:
                    fromPath = prevStates[0][0]
                    temp_cost[binState] = _cost_1
                else:
                    fromPath = prevStates[0][1]
                    temp_cost[binState] = _cost_2

                # print(binState, fromPath, " prev : ", prevStates[0][0], prevStates[0][1], len(self.path[fromPath]))
                temp_path = []
                temp_class = []
                for p in range(len(self.path[fromPath] ) ) :
                    temp_path.append(self.path[fromPath][p])
                    temp_class.append(self.pathClass[fromPath][p])

                temp_path.append(binState)
                temp_class.append(self.states.cluster_class[binState])
                temp_main_path[binState] = temp_path
                temp_main_pathClass[binState] = temp_class
            for state in self.states.states:
                binState = self.states.value[state]
                self.cost[binState] = temp_cost[binState]
                self.path[binState] = temp_main_path[binState]
                self.pathClass[binState] = temp_main_pathClass[binState]
        else:
            for state in self.states.states:
                binState = self.states.value[state]
                mean = self.states.cluster_mean[binState]
                sigma = self.states.cluster_variance[binState]
                # print("true", np.shape(sigma))
                # print(sigma)
                # print(mean)
                _cost = self.GaussProbability(xk, mean, sigma)
                self.cost[binState] = math.log(self.states.probability[binState]) + math.log(_cost)
                self.path[binState] = [binState]
                self.pathClass[binState] = [self.states.cluster_class[binState]]
        # print("Path Length",len(self.path['110']),len(self.pathClass['110']))

    def Viterbi_Distance(self, xk, isFirst):
        temp_cost = {}
        temp_main_path = {}
        temp_main_pathClass = {}
        if not isFirst:
            for state in self.states.states:
                binState = self.states.value[state]
                mean = self.states.cluster_mean[binState]
                val = 0.0
                for i in range(len(xk)):
                    val += (xk[i] - mean[i]) * (xk[i] - mean[i])
                _cost = math.sqrt(val)
                prevStates = self.states.transition[binState]

                _cost_1 = self.cost[prevStates[0][0]] + _cost
                _cost_2 = self.cost[prevStates[0][1]] + _cost

                if _cost_1 < _cost_2:
                    fromPath = prevStates[0][0]
                    temp_cost[binState] = _cost_1
                else:
                    fromPath = prevStates[0][1]
                    temp_cost[binState] = _cost_2

                # print(binState, fromPath, " prev : ", prevStates[0][0], prevStates[0][1], len(self.path[fromPath]))
                temp_path = []
                temp_class = []
                for p in range(len(self.path[fromPath])):
                    temp_path.append(self.path[fromPath][p])
                    temp_class.append(self.pathClass[fromPath][p])

                temp_path.append(binState)
                temp_class.append(self.states.cluster_class[binState])
                temp_main_path[binState] = temp_path
                temp_main_pathClass[binState] = temp_class
            for state in self.states.states:
                binState = self.states.value[state]
                self.cost[binState] = temp_cost[binState]
                self.path[binState] = temp_main_path[binState]
                self.pathClass[binState] = temp_main_pathClass[binState]
        else:
            for state in self.states.states:
                binState = self.states.value[state]
                mean = self.states.cluster_mean[binState]
                val = 0.0
                for i in range(len(xk)):
                    val += (xk[i] - mean[i]) * (xk[i] - mean[i])
                _cost = math.sqrt(val)
                self.cost[binState] = math.log(_cost)
                self.path[binState] = [binState]
                self.pathClass[binState] = [self.states.cluster_class[binState]]


def storeTrainedDataToFile(_states):
    with open('TrainedData.pkl', 'wb') as output:
        pickle.dump(_states, output, pickle.HIGHEST_PROTOCOL)

def getTrainedData():
    with open('TrainedData.pkl', 'rb') as input:
        _states = pickle.load(input)
        return _states

def updateParameters(filepath):
    with open(filepath) as fp:
        n = int(fp.readline())
        line = fp.readline()
        h = [float(x) for x in line.split()]
        line = fp.readline()
        j = [float(x) for x in line.split()]
        mean = j[0]
        variance = j[1]
    return h, mean,variance

def main():
    # random.seed(123)
    # np.random.seed(123)
    isDistanceUsed = True
    parameterFilePath = "Data/param.txt"
    trainFilePath = "Evaluation/train.txt"
    testFilePath = "Evaluation/test.txt"
    h, mean, variance = updateParameters(parameterFilePath)
    print(h, mean,variance)

    n = len(h)
    L = 2
    channel = Channel(h, mean, variance)

    # _states = getTrainedData()

    # '''
    _states = States(n, L)
    _states.initializeStates()
    _states.transitionProb()

    print("Training start...")
    transmission = Transmission(trainFilePath)
    transmission.startTransmission(n,L,channel,_states)

    storeTrainedDataToFile(_states)
    # '''

    print("Testing Start...")
    # x_list = transmission.x
    equilizer = Equilizer(testFilePath,_states,isDistanceUsed)
    equilizer.startEquilizing(n,L,channel)

    '''
    print("\n #########################\n")
    for state in _states.states:
        binState = _states.value[state]
        # print(state)
        print(binState)
        # print(_states.transition[binState])
        print("Mean : \n", _states.cluster_mean[binState] )
        print("Variance : \n",_states.cluster_variance[binState] )
        # print(np.shape(_states.cluster_variance[binState]))
        print("Prob : \n",_states.probability[binState] )
        print("\n")
    # '''

if __name__ == '__main__':
    main()
