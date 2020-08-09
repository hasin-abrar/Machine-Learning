import math
import random
import datetime
import numpy as np
import pandas as pd

def test():
    a = int(13 * 0.6)
    b = int(13 * 0.2)
    c = int(13 * 0.2)
    print(a, b, c)

def identityMatrix():
    x = np.ones((3,3))
    print(x)
    y = np.identity(3)
    y = 3 * y
    print(y, type(y))

def numpyCheck():
    dataFrame = pd.read_csv("Data/dataSmall.csv", header=None)
    dataArray = dataFrame.values
    print(len(dataArray[0]), len(dataFrame.columns))

def testRandom():
    randIndex = random.randint(0, 2)
    print(randIndex)

def checkString():
    x = "ac"+3+"dasd"+str(3)
    print(type(x))
    print(x)

def checkPower():
    x =((3 - 50) ** 2)
    x = 47 **2
    x = math.pow(47,2)
    print(x)

def main():
    # test()
    # numpyCheck()
    # identityMatrix()
    # testRandom()
    # checkPower()
    checkString()
    pass

if __name__ == '__main__':
    main()