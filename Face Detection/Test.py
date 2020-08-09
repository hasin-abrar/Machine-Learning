import shutil

import cv2,os


# f = open('ID archive.txt','w')
f = open('ID archive.txt','r')
i = f.read()
j = int(i)+3
print(j)
f.close()