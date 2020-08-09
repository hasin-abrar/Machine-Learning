import cv2
import numpy as np
import math
import os
import glob
import re
import matplotlib.pyplot as plt

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def createFrames():
    savePath = "Frames/"
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    vidcap = cv2.VideoCapture('movie.mov')
    success, image = vidcap.read()
    _count = 0
    while success:
        cv2.imwrite("Frames/frame_%d.jpg" % _count, image)
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        _count += 1
    print("Frames Created")

def exhaustiveSearch(p):
    mypath = "Frames/"
    savePath = "ExhaustiveSearch/"
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    _count = 0
    frame_searched = 0.0
    total_frames = float(len(glob.glob(mypath +'*.jpg')) )
    # print(total_frames)
    frame = cv2.imread(mypath+"frame_1.jpg")
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    I, J = np.shape(grayFrame)
    ref = cv2.imread('reference.jpg')
    grayRef = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    M, N = np.shape(grayRef)
    indx = [0,0]
    x_start = 0
    x_end = I - M + 1
    y_start = 0
    y_end = J - N + 1
    for f in sorted(glob.glob(mypath + "*.jpg"), key=numericalSort):
        _count += 1
        # print(f)
        frame = cv2.imread(f)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _max = -math.inf
        for i in range(x_start,x_end):
            for j in range(y_start,y_end):
                testSlice = grayFrame[ i: i+M , j : j + N ]
                val = np.multiply(testSlice,grayRef)
                sum_val = np.sum(val)
                frame_searched += 1.0
                if _max < sum_val:
                    _max = sum_val
                    indx[0] = i
                    indx[1] = j
        cv2.rectangle(frame, (indx[1], indx[0]), (indx[1] + N, indx[0] + M), (0, 255, 0), 3)
        cv2.imwrite(savePath+"frame_%d.jpg" % _count, frame)

        x_start = max(0, indx[0] - p)
        x_end = min(I - M + 1, indx[0] + p)
        y_start = max(0, indx[1] - p)
        y_end = min(J - N + 1, indx[1] + p)
    return float(frame_searched / total_frames)



def exhaust(grayFrame, grayRef,indx,p,first = False):
    M,N = np.shape(grayRef)
    I,J = np.shape(grayFrame)
    frame_searched = 0.0
    if first == True:
        x_start = 0
        x_end = I - M + 1
        y_start = 0
        y_end = J - N + 1
    else:
        x_start = max(0, indx[0] - p)
        x_end = min(I - M + 1, indx[0] + p)
        y_start = max(0, indx[1] - p)
        y_end = min(J - N + 1, indx[1] + p)

    t0,t1 = x_start,y_start
    _max = -math.inf
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            sum_val = crossCorrelation(grayFrame, grayRef, i, j, M, N)
            frame_searched += 1.0
            if _max < sum_val:
                _max = sum_val
                t0 = i
                t1 = j

    return t0, t1, frame_searched

def filterAndSample(frame,blur, resize):
    frame = cv2.blur(frame,(blur,blur))
    frame = cv2.resize(frame,None, fx=resize, fy=resize)
    return frame

def hierarchical(p):
    mypath = "Frames/"
    savePath = "HierSearch/"
    given_p = p
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    _count = 0
    frame_searched = 0.0
    total_frames = float(len(glob.glob(mypath + '*.jpg')))
    k = math.ceil(math.log2(p))
    d = int(math.pow(2, k - 1))
    frame = cv2.imread(mypath + "frame_1.jpg")
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    I, J = np.shape(grayFrame)
    refFrame = cv2.imread('reference.jpg')
    grayRef = cv2.cvtColor(refFrame, cv2.COLOR_BGR2GRAY)
    M, N = np.shape(grayRef)
    indx = [0, 0]
    x_start = 0
    x_end = I - M + 1
    y_start = 0
    y_end = J - N + 1
    firstFrame = True
    for f in sorted(glob.glob(mypath + "*.jpg"), key=numericalSort):
        _count += 1
        p = given_p
        # print(f)
        frame = cv2.imread(f)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _max = -math.inf
        _break = True
        if firstFrame:
            indx[0],indx[1], frame_searched_temp = exhaust(grayFrame,grayRef,[],0,first= True)
            frame_searched += frame_searched_temp
            firstFrame = False
        else:
            level = 0
            limit = 3
            x1,y1 = 0,0
            indx_temp = [0,0]
            testFrames = []
            refFrames = []
            testFrames.append(grayFrame)
            refFrames.append(grayRef)
            while True:
                level += 1
                if level == limit:
                    break
                testTemp = filterAndSample(testFrames[-1],2,0.5)
                refTemp = filterAndSample(refFrames[-1],2,0.5)
                testFrames.append(testTemp)
                refFrames.append(refTemp)
            while True:
                level -= 1
                testTemp = testFrames.pop(-1)
                refTemp = refFrames.pop(-1)
                if level == limit - 1 :
                    div = math.pow(2, level)
                    indx_temp[0] = math.ceil(indx[0] / div)
                    indx_temp[1] = math.ceil(indx[1] / div)
                    p_temp = math.ceil((p / div))
                else:
                    p_temp = 1
                    indx_temp[0] = math.ceil(indx[0]/math.pow(2,level) ) + 2*x1
                    indx_temp[1] = math.ceil(indx[1] / math.pow(2, level)) + 2 * y1
                # print(p_temp, indx_temp,math.ceil(indx[0]/math.pow(2,level) ),math.ceil(indx[1]/math.pow(2,level) ) )
                x1, y1, frame_searched_temp = exhaust(testTemp, refTemp,indx_temp, p_temp)
                frame_searched += frame_searched_temp
                # print(x1,y1)

                if level == 0:
                    indx[0] = x1
                    indx[1] = y1
                    break
                else:
                    x1 = x1 - math.ceil(indx[0]/math.pow(2,level) )
                    y1 = y1 - math.ceil(indx[1]/math.pow(2,level) )

        cv2.rectangle(frame, (indx[1], indx[0]), (indx[1] + N, indx[0] + M), (0, 0, 255), 3)
        cv2.imwrite(savePath + "frame_%d.jpg" % _count, frame)
    return float(frame_searched / total_frames)


def crossCorrelation(grayFrame,grayRef,i,j,M,N):
    testSlice = grayFrame[i: i + M, j: j + N]
    val = np.multiply(testSlice, grayRef)
    sum_val = np.sum(val)
    return sum_val

def logSearch(p):
    mypath = "Frames/"
    savePath = "LogSearch/"
    given_p = p
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    modified_images = []
    _count = 0
    frame_searched = 0.0
    total_frames = float(len(glob.glob(mypath + '*.jpg')))
    k = math.ceil(math.log2(p))
    d = int(math.pow(2, k - 1))
    frame = cv2.imread(mypath + "frame_1.jpg")
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    I, J = np.shape(grayFrame)
    ref = cv2.imread('reference.jpg')
    grayRef = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    M, N = np.shape(grayRef)
    indx = [0, 0]
    x_start = 0
    x_end = I - M + 1
    y_start = 0
    y_end = J - N + 1
    firstFrame = True
    for f in sorted(glob.glob(mypath + "*.jpg"), key=numericalSort):
        _count += 1
        p = given_p
        # print(f)
        frame = cv2.imread(f)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _max = -math.inf
        _break = True
        if firstFrame:
            for i in range(x_start, x_end):
                for j in range(y_start, y_end):
                    sum_val = crossCorrelation(grayFrame,grayRef,i,j,M,N)
                    frame_searched += 1.0
                    if _max < sum_val:
                        _max = sum_val
                        indx[0] = i
                        indx[1] = j
            firstFrame = False
        else:
            # print(indx[0],indx[1])
            sum_val = crossCorrelation(grayFrame, grayRef, indx[0], indx[1], M, N)
            frame_searched += 1.0
            if _max < sum_val:
                _max = sum_val
            while True:
                k = math.ceil(math.log2(p))
                d = int(math.pow(2, k - 1))
                p = math.ceil(p / 2)
                if d == 0:
                    break
                x_start = max(0, indx[0] - p)
                x_end = min(I - M + 1, indx[0] + p)
                y_start = max(0, indx[1] - p)
                y_end = min(J - N + 1, indx[1] + p)
                for i in range(x_start, x_end + 1, d ):
                    for j in range(y_start, y_end + 1, d ):
                        sum_val = crossCorrelation(grayFrame, grayRef, i, j, M, N)
                        frame_searched += 1.0
                        # print(i,j)
                        if _max < sum_val:
                            _max = sum_val
                            indx[0] = i
                            indx[1] = j

        cv2.rectangle(frame, (indx[1], indx[0]), (indx[1] + N, indx[0] + M), (0, 0, 255), 3)
        cv2.imwrite(savePath + "frame_%d.jpg" % _count, frame)
    return float(frame_searched / total_frames)

def makeVideo(mypath, videoName):
    # mypath = "Frames/"
    frame = cv2.imread(mypath+"frame_1.jpg")
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width= np.shape(frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(videoName+'.mov', fourcc,60.0, (width, height))

    for infile in sorted(glob.glob(mypath+"*.jpg"), key=numericalSort):
        frame = cv2.imread(infile)
        out.write(frame)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    out.release()
    # cv2.destroyAllWindows()

def plot(x, y, xLabel, yLabel, title):
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel(xLabel)
    # naming the y axis
    plt.ylabel(yLabel)

    # giving a title to my graph
    plt.title(title)

    # function to show the plot
    plt.show()


def main():
    P = [1, 2, 4, 8, 16, 32, 64]
    exhaust_avg = []
    hier_avg = []
    log_avg = []
    exhaust = "ExhaustiveSearch/"
    logS = "LogSearch/"
    hier = "HierSearch/"
    # createFrames()

    for p in P:
        print("P: ", p, " ExhaustSearch...")
        x = exhaustiveSearch(p)
        exhaust_avg.append(x)
        print("P: ",p," LogSearch...")
        y = logSearch(p)
        log_avg.append(y)
        print("P: ", p, " HierSearch...")
        z = hierarchical(p)
        hier_avg.append(z)
        # makeVideo(exhaust, "Exhaust_"+str(p))
        # makeVideo(logS, "LogSearch_"+str(p))
        # makeVideo(hier, "Hier_"+str(p))
    print(exhaust_avg)
    print(log_avg)
    print(hier_avg)
    plot(P, exhaust_avg, "P", "Avg no. of frames", "Exhaustive")
    plot(P, log_avg, "P", "Avg no. of frames", "LogSearch")
    plot(P, hier_avg, "P", "Avg no. of frames", "HierSearch")

if __name__ == '__main__':
    main()