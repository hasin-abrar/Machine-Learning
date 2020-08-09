import cv2,os

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Id = input("Enter your ID : ")

f = open('ID archive.txt','r')
i = f.read()
f.close()
Id = int(i)
nextId = Id +1
f = open('ID archive.txt','w')
f.write(str(nextId))
f.close()

sampleNum = 0
newPathFile = "FoundImages/"+str(Id)+"/"
if not os.path.exists(newPathFile):
    os.makedirs(newPathFile)

while True :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces :
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum = sampleNum + 1
        snStr = str(sampleNum)
        print("Type of snStr : ")
        type(snStr)
        print("\nvalue : "+snStr)
        path = "dataset/User."+str(Id)+"."+snStr+".jpg"
        type(path)
        print(path)
        cv2.imwrite(path,gray[y:y+h,x:x+w])
        # cv2.imwrite("dataSet/User." + Id + '.',sampleNum,".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Frame',img)

    if(cv2.waitKey(100) & 0xFF == ord('q')):
        break
    elif sampleNum > 50:
        break

cam.release()
cv2.destroyAllWindows()

