import shutil

import cv2,os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_SIMPLEX


def getRecognizedImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    foundID = []
    _count = 1
    for imgPath in imagePaths:
        print(str(_count) +" : "+ imgPath)
        # print(imgPath)
        if os.path.split(imgPath)[-1].split(".")[-1] != "jpg" and os.path.split(imgPath)[-1].split(".")[-1] != "JPG" :
            continue
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if os.path.isfile(imgPath) and conf>50:
                if Id == 1:
                    nameId = "Antu"
                elif Id == 4:
                    nameId= "Shaon"
                elif Id == 5:
                    nameId= "Chashi"
                elif Id == 6:
                    nameId= "Sagor"
                elif Id == 7:
                    nameId= "Angela"
                foundID.append(Id)
                cv2.putText(img, nameId, (x, y + h), font, 1, (0, 255, 0))
            # ************FILE NAME ALADA KORE STORE KORA&*********************

                # shutil.copy2(imgPath, "FoundImages/"+str(Id)+"/")
                # os.remove(imgPath)
                #
                # if(Id == 1):
                #     shutil.copy2(imgPath, "FoundImages/1/")
                #     os.remove(imgPath)
                # elif Id == 2:
                #     shutil.copy2(imgPath, "FoundImages/2/")
                #     os.remove(imgPath)
        cv2.imshow('Frame', img)
        cv2.waitKey(2000)

        foundID.sort()
        isGroupPhoto = True
        for _id in findID:
            try:
                _id = foundID.index(_id)
            except ValueError:
                isGroupPhoto = False
                break
        if isGroupPhoto:
            print("Group Photo")
            shutil.copy2(imgPath, newPathFile)
        else:
            print("NOT a group Photo")

        print(foundID)

        print()
        foundID[:] = []

# START
findID = [6,1]
findID.sort()
print(findID)
folderName = ""
for i in findID:
    folderName += str(i)

newPathFile = "FoundImages/Group Photo/"+folderName+"/"
if not os.path.exists(newPathFile):
    os.makedirs(newPathFile)
getRecognizedImages('SourceImages/')
# getRecognizedImages('F:\Downloads\DATASET')