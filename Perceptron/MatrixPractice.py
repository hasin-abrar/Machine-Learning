import numpy as np
import pandas as pd
import math

# x = M X 1 that is need to transposed before sending
def basicPerceptron(examples,x,feature_number):
    weight = np.zeros((feature_number+1, 1))
    _count = 0
    # print(weight)
    while(True):
        _count += 1
        # if(_count > 1000) :
        #     break
        Y = []
        delta = 0
        hyper_parameter = 1
        for i in range(len(x)):
            if examples[i][-1] == 1:
                delta = -1
            else:
                delta = 1
            fea_val = np.array([x[i]]).transpose()
            # print("fea_val ",fea_val)
            mult = weight.transpose().dot(fea_val)
            # print("mult ",mult)
            check = delta*mult
            # print("check ",check)
            if check >= 0 :
                Y.append(i)
        # print(Y)
        val = np.zeros( (feature_number+1,1))
        for i in Y:
            if examples[i][-1] == 1:
                delta = -1
            else:
                delta = 1
            fea_val = np.array([x[i]]).transpose()
            val += delta*fea_val  # scalar multiplication
        # print("val ",val)
        val *= hyper_parameter
        # print("val ", val)
        weight = weight - val
        # print("weight ",weight)
        if len(Y) == 0:
            print(_count)
            break
    return weight


def prediction(weight,x):
    check = weight.transpose().dot(x)
    # print(check)
    if check > 0:
        return 1
    else:
        return 2

# weight is a list of numpy
def predictionMultiClass(weight, x, classCount):
    _max = -math.inf
    foundClass = 0
    for i in range(classCount):
        check = weight[i].transpose().dot(x)
        if(check > _max):
            _max = check
            foundClass = i + 1
    return foundClass


def rewardAndPunishment(x,y, feature_number):
    total = len(x)
    iteration = 0
    parameter = 1
    weight = np.zeros((feature_number + 1, 1))
    while(True):
        iteration+=1
        _count = 0
        for i in range(total):
            x_val = np.array( [ x[i] ] ).transpose()
            val = x_val * parameter
            check = weight.transpose().dot(x_val)
            if y[i] == 1 and check <= 0:
                weight += val
            elif y[i] == 2 and check >= 0:
                weight -= val
            else:
                _count += 1
        if _count == total:
            print(iteration)
            break
        if iteration > 1000:
            break
    return weight

def pocketPerceptron(x,y,feature_number):
    weight = np.zeros((feature_number + 1, 1))
    _count = 0
    weight_stored = np.zeros((feature_number + 1, 1))
    hs = 0
    while (True):
        _count += 1
        if(_count > 500) :
            break
        Y = []
        delta = 0
        hyper_parameter = 1
        for i in range(len(x)):
            if y[i] == 1:
                delta = -1
            else:
                delta = 1
            fea_val = np.array([x[i]]).transpose()
            mult = weight.transpose().dot(fea_val)
            check = delta * mult
            if check >= 0:
                Y.append(i)
        # print(Y)
        val = np.zeros((feature_number + 1, 1))
        for i in Y:
            if y[i] == 1:
                delta = -1
            else:
                delta = 1
            fea_val = np.array([x[i]]).transpose()
            val += delta * fea_val  # scalar multiplication
        val *= hyper_parameter
        weight = weight - val
        h = 0

        for i in range(len(x) ):
            sample = np.array( [ x[i] ] ).transpose()
            if prediction(weight,sample) == y[i]:
                h += 1
        if h >= hs:
            hs = h
            weight_stored = np.copy(weight)
            if hs == len(x) :
                print(_count)
                break
        # if len(Y) == 0:
        #     print(_count)
        #     break
    return weight_stored


def keslerPerceptron(x,y,classCount, feature_number):
    total = len(x)
    iteration = 0
    parameter = 1
    weight = []
    for i in range(classCount):
        weight.append( np.zeros((feature_number + 1, 1)) )
    while (True):
        iteration += 1
        _count = 0
        for i in range(total):
            x_val = np.array([x[i]]).transpose()
            val = x_val * parameter
            foundClass = 0
            _maxVal = -math.inf
            for c in range(classCount):
                weightVector = weight[c]
                check = weightVector.transpose().dot(x_val)
                if check > _maxVal :
                    _maxVal = check
                    foundClass = c + 1 # better to store a dictionary for class type
            actualClass = y[i]
            if actualClass != foundClass:
                weightVector = weight[foundClass - 1]
                weightVector -= val
                weightVector = weight[actualClass - 1]
                weightVector += val
            else:
                _count += 1
        if _count == total:
            print(iteration)
            break
        if iteration > 1000:
            break
    return weight


# dataset = pd.read_csv("Binary/binaryTrain.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/trainLinearlySeparable.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/trainLinearlyNonSeparable.csv",header = None)
dataset = pd.read_csv("Data/kesler/Train.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/evaluation/trainLinearlySeparable.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/evaluation/trainLinearlyNonSeparable.csv",header = None)
_x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
examples = dataset.iloc[:,:].values
feature_count = 3
# print(len(examples))
x = np.zeros( (len(_x),feature_count + 1) )
for i in range(len(_x)):
    x[i] = np.append(_x[i],1)

# weight = basicPerceptron(examples,x,feature_count)
# weight = rewardAndPunishment(x,y,feature_count)
# weight = pocketPerceptron(x,y,feature_count)
# print(weight)

weightList = keslerPerceptron(x,y,3,feature_count)
print(weightList)



# dataset = pd.read_csv("Data/perceptron/testLinearlySeparable.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/testLinearlyNonSeparable.csv",header = None)
dataset = pd.read_csv("Data/kesler/Test.csv",header = None)
# dataset = pd.read_csv("Offline/Test.csv")
# dataset = pd.read_csv("Data/perceptron/evaluation/testLinearlySeparable.csv",header = None)
# dataset = pd.read_csv("Data/perceptron/evaluation/testLinearlyNonSeparable.csv",header = None)
_x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
examples = dataset.iloc[:,:].values
y = np.array(y)
x = np.zeros( (len(_x),feature_count + 1) )
for i in range(len(_x)):
    x[i] = np.append(_x[i],1)
index = 0
match = 0
not_match = 0
for sam in x:
    sample = np.array([sam]).transpose()
    # found = prediction(weight,sample)
    found = predictionMultiClass(weightList,sample,classCount=3)
    if y[index] == found :
        match += 1
    else:
        print(index+2,found, y[index])
        not_match += 1
        # print(found, y[index],match,not_match)
    index += 1

print(match, not_match)
accuracy = match/(match + not_match)
print("accuracy : ",accuracy*100)

