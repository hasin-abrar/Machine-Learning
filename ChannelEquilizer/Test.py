import  numpy as np
import pickle

def test():
    n = 5
    res = "{0:b}".format(n)
    print(res[0], res[1])

def intDiv():
    a = int(5.8/2)
    a = 5.9//2
    print(a)

def fileOperation():
    filepath = 'Data/test.txt'
    with open(filepath) as fp:
        line = fp.readline()
        for i in range(3):
            line += "addThis"
    print(line)
    print(line[-1])
        # cnt = 1
        # while line:
        #     # print("Line {}: {}".format(cnt, line.strip()))
        #     # print(line)
        #     print(line[0])
        #     print(len(line))
        #     line = fp.readline()
        #     cnt += 1

def normalRandom():
    mu = 10
    sigma = 0.5 # mean and standard deviation
    x = np.random.normal(mu, sigma,2)
    print(x)

def listToString():
    a = ['1','2']
    b = ''.join(a)
    print(b)

def numpyCheck():
    a = [1,2]
    b = [2,3]
    c = [a,b]
    c = np.array([a])
    # d = np.mean(c,axis=1)
    print(c)
    d = np.transpose(c)
    print(np.shape(d),len(c),np.shape(c) )

def listCheck():
    a = [1,2]
    b = list(reversed(a))
    print(b)

def storeFile():
    with open('numpyStored.pkl', 'wb') as output:
        x = [3,1,22,3]
        x = np.array(x)
        print(x)
        pickle.dump(x, output, pickle.HIGHEST_PROTOCOL)

        y = [[12,1],[3,4]]
        y = np.array(y)
        print(y)
        pickle.dump(y, output, pickle.HIGHEST_PROTOCOL)

def getObject():
    with open('numpyStored.pkl', 'rb') as input:
        x = pickle.load(input)
        print(x)

        y = pickle.load(input)
        print(y)

def getGaussian(xvect, mewk, sigmak,l):

    xi = np.zeros((l,1))
    for i in range(len(xi)):
        xi[i][0] = xvect[i]
    det_sigmak = np.linalg.det(sigmak)
    det_sigmak = np.absolute(det_sigmak)
    inv_sigmak = np.linalg.inv(sigmak)
    constant = 1.0 / (np.sqrt((np.power(2*np.pi,l)) * det_sigmak))
    xi_min_mewk = np.subtract(xi,mewk)

    temp = np.dot(np.transpose(xi_min_mewk),inv_sigmak)

    exp_val = np.dot(temp,xi_min_mewk)
    exp_val = -0.5 * exp_val
    if(exp_val < -500):
        exp_val = -500
    elif (exp_val > 500):
        exp_val = 500
    ans = constant*np.exp(exp_val)

    return ans

def main():
    # test()
    # intDiv()
    # fileOperation()
    # normalRandom()
    # listToString()
    # numpyCheck()
    # listCheck()
    # storeFile()
    getObject()
    pass
if __name__ == '__main__':
    main()