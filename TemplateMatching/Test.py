import cv2
import numpy as np
import math
import glob
import os
import re
import matplotlib.pyplot as plt

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def cameraShow():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_Camera.avi', fourcc, 20.0, (640, 480),0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def showFiles():
    mypath = "Frames/"
    frame = cv2.imread(mypath+"frame_1.jpg")
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width= np.shape(frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_DONE.mov', fourcc,60.0, (width, height))

    for infile in sorted(glob.glob(mypath+"*.jpg"), key=numericalSort):
        frame = cv2.imread(infile)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    # files = [f for f in os.listdir(pathIn) if os.isfile(join(pathIn, f))]
    #
    # # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    files = []
    for infile in sorted(glob.glob(pathIn + "*.jpg"), key=numericalSort):
        files.append(infile)

    for i in range(len(files)):
        filename = files[i]
        # reading each files
        img = cv2.imread(filename)
        # height, width, layers = np.shape(img)
        size = (640, 480)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def makeVideo():
    # pathIn = './final_out/'
    # pathOut = 'video.mov'
    # fps = 0.5
    # convert_frames_to_video(pathIn, pathOut, fps)

    pathIn = './Frames/'
    pathOut = 'video.mov'
    fps = 50
    convert_frames_to_video(pathIn, pathOut, fps)


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


def checkImage():
    frame = cv2.imread("Frames/frame_1.jpg")
    ref = cv2.imread('reference.jpg')
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayRef = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    print(np.shape(grayFrame))
    print(np.shape(grayRef))
    x,y = np.shape(grayRef)
    print(x,y)
    while True:
        cv2.imshow('Ref',grayRef)
        cv2.imshow('GrayFrame',grayFrame)

        if cv2.waitKey(0):
            break

def rectangleCheck():
    frame = cv2.imread("Frames/frame_1.jpg")
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(grayFrame,(10,30),(200,300),(0,255,0),3)
    while True:
        cv2.imshow("Sample",grayFrame)
        if cv2.waitKey(0):
            break


def exhaustiveSearch():
    frame = cv2.imread("Frames/frame_100.jpg")
    ref = cv2.imread('reference.jpg')
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayRef = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    I,J = np.shape(grayFrame)
    M,N = np.shape(grayRef)
    _max = -math.inf
    _break = True
    indx = [0,0]
    for i in range(I - M + 1):
        for j in range(J - N + 1):
            testSlice = grayFrame[ i: i+M , j : j + N ]
            # cv2.imshow("Slice",testSlice)
            if _break:
                print(np.shape(testSlice))
                _break = False
            val = np.multiply(testSlice,grayRef)
            sum_val = np.sum(val)
            if _max < sum_val:
                _max = sum_val
                indx[0] = i
                indx[1] = j
    print(indx)
    while True:
        testSlice = grayFrame[indx[0]: indx[0] + M, indx[1]: indx[1] + N]
        cv2.rectangle(grayFrame, (indx[1],indx[0]), (indx[1] + N, indx[0] + M), (0, 255, 0), 1)
        cv2.imshow("Slice_Main", testSlice)
        cv2.imshow("MAIN", grayFrame)
        cv2.imshow("ref",grayRef)
        if cv2.waitKey(0):
            break

def checkSum():
    x = [[1,1],
         [2,3],
         [4,5]
         ]
    np_x = np.array(x)
    print(np.sum(np_x))

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
    # for f in os.listdir(mypath):
    # for f in glob.glob(mypath+"*.jpg"):
    for f in sorted(glob.glob(mypath + "*.jpg"), key=numericalSort):
        _count += 1
        p = given_p
        print(f)
        frame = cv2.imread(f)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _max = -math.inf
        _break = True
        if firstFrame:
            for i in range(x_start, x_end):
                for j in range(y_start, y_end):
                    sum_val = crossCorrelation(grayFrame,grayRef,i,j,M,N)
                    if _max < sum_val:
                        _max = sum_val
                        indx[0] = i
                        indx[1] = j
            firstFrame = False
        else:
            # print(indx[0],indx[1])
            sum_val = crossCorrelation(grayFrame, grayRef, indx[0], indx[1], M, N)
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
                        # print(i,j)
                        if _max < sum_val:
                            _max = sum_val
                            indx[0] = i
                            indx[1] = j
        # if _break and _count == 2:
        #     break

        cv2.rectangle(frame, (indx[1], indx[0]), (indx[1] + N, indx[0] + M), (0, 0, 255), 3)
        cv2.imwrite(savePath + "frame_%d.jpg" % _count, frame)
        print(indx[1],indx[0])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     continue

    # cv2.destroyAllWindows()

def exhaust(grayFrame, grayRef,indx,p,first = False):
    M,N = np.shape(grayRef)
    I,J = np.shape(grayFrame)

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

            if _max < sum_val:
                _max = sum_val
                t0 = i
                t1 = j

    return t0, t1


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
        print(f)
        frame = cv2.imread(f)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _max = -math.inf
        _break = True
        if firstFrame:
            indx[0],indx[1] = exhaust(grayFrame,grayRef,[],0,first= True)
            # for i in range(x_start, x_end):
            #     for j in range(y_start, y_end):
            #         sum_val = crossCorrelation(grayFrame, grayRef, i, j, M, N)
            #         if _max < sum_val:
            #             _max = sum_val
            #             indx[0] = i
            #             indx[1] = j
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
                # print(np.shape(refTemp))
                testFrames.append(testTemp)
                refFrames.append(refTemp)
            # print(len(testFrames))
            # for i in range(limit):
            #     print(np.shape(testFrames[i]))
            #     print(np.shape(refFrames[i]))
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
                x1, y1 = exhaust(testTemp, refTemp,indx_temp, p_temp)
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
        print(indx[1], indx[0])

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

def resize():
    # path = "Frames/frame_1.jpg"
    path = "reference.jpg"
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print(np.shape(frame))
    frame = cv2.blur(frame,(2,2))
    print(np.shape(frame))
    frame = cv2.resize(frame, None, fx=0.5,fy=0.5)
    print(np.shape(frame))
    while True:
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    # hierarchical(7)
    showFiles()


if __name__ == '__main__':
    main()