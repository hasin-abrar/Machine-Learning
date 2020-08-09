import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def showClusteredData(df,K):
    data_array = np.array(df)
    dimension = len(df.columns)
    data_mean = np.mean(data_array,axis=0,dtype=float) # make sure if it is okay
    data_covar = np.cov(data_array)
    eigenvalue, eigenvector = np.linalg.eig(data_covar)
    index_max_1 = np.argmax(eigenvalue)
    eigenvalue = np.delete(eigenvalue,index_max_1)
    index_max_2 = np.argmax(eigenvalue)
    principal_components = np.empty([K,dimension])

def check():
    s = np.array([[2, 1, 2],
                  [1, 2, 3],
                  [1, 2, 3],
                  [3, 1, 5]],dtype=float)
    s[1] = np.multiply(s[1],0.2)
    print(s)

def matplot():


    # Fixing random state for reproducibility
    np.random.seed(19680801)

    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii

    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()

def matplot2():
    X = [1, 2, 3, 4]
    x2 = [2,3,4,5]
    Y1 = [4, 8, 12, 16]
    Y2 = [1, 4, 9, 16]

    plt.scatter(X, Y1, color='red')
    plt.scatter(x2, Y2, color='blue')
    plt.show()

def main():
    # df = pd.read_csv('data_sh.txt', delimiter='\t', header=None)
    # v = np.empty([2,3])
    # s = np.array([[2,1,2],
    #               [1,2,3],
    #               [1,2,3],
    #               [3,1,5]])
    # row = [1,2]
    # p = s[row]
    # print(p)
    # check()
    matplot2()

if __name__ == '__main__':
    main()



