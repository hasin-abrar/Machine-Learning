import math
import pickle
import random
import time
import numpy as np
import pandas as pd

class Model:
    def __init__(self, k, lambda_u, lambda_v):
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.u = {}
        self.v = {}
        self.rmse = 0.0

def importData():
    # dataFrame = pd.read_csv("Data/small.csv", header = None)
    dataFrame = pd.read_csv("Data/data.csv", header = None)
    dataArray = dataFrame.values
    noOfItems = len(dataFrame.columns)
    # print(noOfItems)
    # dataFrame = dataFrame.drop(dataFrame.columns[0], axis=1)
    userCount = len(dataFrame)
    train_list = {}
    trainFull_list = {}
    validation_list = {}
    test_list = {}
    for i in range(userCount):
        voted = dataArray[i][0]
        train_count = int(voted * 0.6)
        validation_count = int(voted * 0.2)
        test_count = int(voted * 0.2)
        diff = voted - (train_count + validation_count + test_count)
        while diff > 0:
            randIndex = random.randint(0,2)
            if randIndex == 0:
                train_count += 1
            elif randIndex == 1:
                validation_count += 1
            else:
                test_count += 1
            diff -= 1

        availableIndexList = []
        # print(train_count,validation_count,test_count)
        for j in range(1,noOfItems):
            if dataArray[i][j] != 99.0:
                availableIndexList.append(j)
        ################# ###################
        # print("avail len : ",len(availableIndexList))
        train_list[i] = []
        trainFull_list[i] = []
        validation_list[i] = []
        test_list[i] = []
        trainIndexList = []
        validationIndexList = []
        testIndexList = []
        while train_count > 0:
            indx = random.choice(availableIndexList)
            trainIndexList.append(indx)
            availableIndexList.remove(indx)
            train_count -= 1
        while validation_count > 0:
            indx = random.choice(availableIndexList)
            validationIndexList.append(indx)
            availableIndexList.remove(indx)
            validation_count -= 1
        while test_count > 0:
            indx = random.choice(availableIndexList)
            testIndexList.append(indx)
            availableIndexList.remove(indx)
            test_count -= 1
        # print("avain idx length now :",len(availableIndexList))
        trainSet = set(trainIndexList)
        validationSet = set(validationIndexList)
        testSet = set(testIndexList)
        for j in range(1,noOfItems):
            if j in trainSet:
                train_list[i].append(dataArray[i][j])
            else:
                train_list[i].append(99.0)
            if j in validationSet:
                validation_list[i].append(dataArray[i][j])
            else:
                validation_list[i].append(99.0)
            if j in testSet:
                test_list[i].append(dataArray[i][j])
                trainFull_list[i].append(99.0)
            else:
                test_list[i].append(99.0)
                trainFull_list[i].append(dataArray[i][j])

    return train_list, validation_list, trainFull_list, test_list, userCount, noOfItems


def getPrediction(i, j, model):
    sum = 0.0
    K = model.k
    # print(i, model)
    u_vect = model.u[i] # 1 X k
    v_vect = model.v[j] # k X 1
    for k in range(K):
        # print(k)
        # y = u_vect[0]
        # x = u_vect[0][k]
        # print(x,y, np.shape(u_vect),k)
        sum += u_vect[0][k] * v_vect[k][0]
    return sum

def ALS(model, train_list, userCount, itemCount):
    k = model.k
    lambda_u = model.lambda_u
    lambda_v = model.lambda_v
    tempModel = Model(model.k,model.lambda_u, model.lambda_v)
    prevDiff = math.inf
    loopCount = 0
    u = {} # always gives the TRANSPOSED result 1 X k
    v = {} # k X 1
    for i in range(userCount):
        u[i] = np.random.rand(1,k) # 1 X k ( init as transposed )
        # v[i] = np.zeros((k,1)) # k X 1 ( as should be )
        # for j in range(k):
        #     print(temp_u[i][j])
        #     u[i][j] = temp_u[i][j]
        # print(u[i])
    '''
    for i in range(userCount):
        for j in range(k):
            print(temp_u[i][j])
            u[i][0][j] = temp_u[i][j]
        print(u[i])
    '''
    print("#######")
    while True:
        loopCount += 1
        for i in range(0,itemCount-1):
            sum = np.zeros((k,k))
            sum2 = np.zeros((k,1))
            for j in range(userCount):
                if train_list[j][i] != 99.0:
                    temp = np.matmul(np.transpose(u[j]), u[j]) # reverse style as u[j] is the transposed version
                    sum = np.add(temp,sum) # k X k
                    temp2 = train_list[j][i] * np.transpose(u[j])
                    sum2 = np.add(sum2, temp2) # k X 1
            # print(sum)
            # print(sum2)
            temp3 = lambda_v * np.identity(k)
            sum = np.add(temp3, sum) # k X k
            inverseSum = np.linalg.inv(sum)
            v[i] = np.matmul(inverseSum, sum2) # k X 1
        for i in range(userCount):
            sum = np.zeros((k, k))
            sum2 = np.zeros((k, 1))
            for j in range(0,itemCount-1):
                if train_list[i][j] != 99.0:
                    temp = np.matmul(v[j],np.transpose(v[j]))
                    sum = np.add(sum , temp) # k X k
                    temp2 = train_list[i][j] * v[j]
                    sum2 = np.add(sum2, temp2)
            temp3 = lambda_u * np.identity(k)
            sum = np.add(sum, temp3)
            inverseSum = np.linalg.inv(sum)
            u[i] = np.matmul(inverseSum, sum2) # k X 1
            u[i] = np.transpose(u[i]) # 1 X k ( as was init at the first place)
        tempModel.u = u
        tempModel.v = v
        '''
        for i in range(userCount):
            print(u[i])
        for i in range(itemCount-1):
            print(v[i])
        '''
        diff = getRMSE(userCount, itemCount, train_list, tempModel)
        # print(k,"#",loopCount,":",diff)
        if (math.fabs(diff - prevDiff) <= 0.01 and loopCount>1) or loopCount > 100:
            if loopCount > 100:
                model = tempModel
            printString = "k= " + str(model.k) + " lambda_u = " + str(model.lambda_u) \
                          + " lambda_v = " + str(model.lambda_v) + " training rmse : " + str(diff)
            reportToFile(printString)
            break
        prevDiff = diff
        model = tempModel

    # print(model.k,model.u[0])
    return model


def getRMSE(userCount,itemCount, item_list, model):
    # RMSE
    rmse = 0.0
    _count = 0
    for i in range(userCount):
        for j in range(0,itemCount-1):
            if item_list[i][j] != 99.0:
                rmse += (math.pow(item_list[i][j] - getPrediction(i, j, model), 2) )
                _count += 1
    rmse /= _count
    rmse = math.sqrt(rmse)
    return rmse

def reportToFile(printString):
    with open('report_Online.txt', 'a') as file:
        file.write(printString)
        file.write("\n\n")
    print(printString)

def gridSearch(train_list,validation_list,trainFull_list, userCount, itemCount):
    K = [5]
    _lambda = [0.01, 0.1,1.0,10.0]
    _max_rmse = math.inf
    final_model = Model(K[0],10.0,1.0)
    _break = False
    # for k in K:
        # for lambda_u in _lambda:
        #     for lambda_v in _lambda:
    k = 5
    lambda_u = 10.0
    lambda_v = 1.0
    model = Model(k, lambda_u, lambda_v)
    model = ALS(model, train_list, userCount, itemCount)
    model.rmse = getRMSE(userCount, itemCount, validation_list,model)
    printString = "k= "+str(model.k)+ " lambda_u = "+ str(model.lambda_u)\
                  + " lambda_v = "+str(model.lambda_v)+" validation rmse : "+str(model.rmse)
    # print(type(printString))
    reportToFile(printString)
    # print("k=", model.k, "lambda_u =", model.lambda_u, "lambda_v =", model.lambda_v,
    #       "validation rmse :",model.rmse)
    if model.rmse < _max_rmse:
        _max_rmse = model.rmse
        final_model = model
    # if _break:
    #     break
    printString = "Best on validation : "+str(final_model.k)+" "+str(final_model.lambda_u)+" "\
                  + str(final_model.lambda_v)+" "+str(final_model.rmse)
    reportToFile(printString)
    # print("Best on validation :",final_model.k, final_model.lambda_u, final_model.lambda_v,
    #             final_model.rmse)
    final_model = ALS(final_model,trainFull_list,userCount,itemCount)
    return final_model

def recommendationEngine(model,test_list,userCount,itemCount):
    # RMSE
    # diff = 0.0
    # for i in range(userCount):
    #     for j in range(itemCount):
    #         if test_list[i][j] != 99:
    #             diff += ((test_list[i][j] - getPrediction(i, j, model)) ** 2)
    # diff /= userCount
    # diff = math.sqrt(diff)
    model.rmse = getRMSE(userCount,itemCount, test_list,model)
    printString = "TEST DATA: k= "+str(model.k)+ " lambda_u = "+str(model.lambda_u)+ " lambda_v = "\
                  +str(model.lambda_v)+ " RMSE : "+str(model.rmse)
    reportToFile(printString)
    # print("TEST DATA: k=", model.k, "lambda_u =", model.lambda_u, "lambda_v =", model.lambda_v, "RMSE :",
    #       model.rmse)
    return model.rmse


def printImportedData(train_list, validation_list, trainFull_list, test_list, userCount, itemCount):
    print()
    for i in range(userCount):
        print(train_list[i])
        print(validation_list[i])
        print(trainFull_list[i])
        print(test_list[i])
        print()

def storeTrainedDataToFile(model):
    with open('TrainedData.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def getTrainedData():
    with open('TrainedData.pkl', 'rb') as input:
        model = pickle.load(input)
        return model


def main():
    np.random.seed(1)
    random.seed(1)
    start_time = time.time()
    train_list, validation_list, trainFull_list, test_list, userCount, itemCount = importData()
    import_time = time.time()
    # printImportedData(train_list, validation_list, trainFull_list, test_list, userCount, itemCount)
    # '''

    model = gridSearch(train_list, validation_list, trainFull_list, userCount, itemCount)
    storeTrainedDataToFile(model)
    train_time = time.time()

    model = getTrainedData()
    RMSE = recommendationEngine(model,test_list,userCount,itemCount)
    finish_time = time.time()
    print("TEST data RMSE =",RMSE)
    print("import time:",(import_time - start_time))
    print("training time:",train_time - import_time)
    print("testing time:",finish_time - train_time)
    print("Total time:",finish_time - start_time)
    # '''


if __name__ == '__main__':
    main()