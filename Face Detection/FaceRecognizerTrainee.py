import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print("Came and watched")
    faceSamples = []
    Ids = []

    for imgPath in imagePaths:

        if(os.path.split(imgPath)[-1].split(".")[-1] != "jpg"):
            continue
        # print("Now hwo many")
        pilImage = Image.open(imgPath).convert('L')
        imageNP = np.array(pilImage,'uint8')
        Id = int(os.path.split(imgPath)[-1].split(".")[1])

        faces = detector.detectMultiScale(imageNP)
        for(x,y,w,h) in faces:
            faceSamples.append(imageNP[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('F:\Downloads\DATASET\dataset\dataset')
recognizer.train(faces,np.array(Ids))
recognizer.save('trainer/trainer.yml')