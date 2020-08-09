import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (225, 0,0), 2 )
        Id,conf = recognizer.predict(gray[y:y+h, x:x+w])
        print(Id)
        print(conf)
        if conf > 50:
            if (Id == 1):
                Id = "Antu"
            elif Id == 4:
                Id = "Shaon"
            elif Id == 5:
                Id = "Chashi"
            elif Id == 6:
                Id = "Sagor"
            elif Id == 7:
                Id = "Angela"

        else:
            Id = "chini na tomare"
        cv2.putText(img, Id, (x,y+h), font, 1, (0,255,0))
    cv2.imshow('img',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()