import cv2 as cv
import numpy as np

shw = cv.imshow
minArea =500
frameWidth,widthImg = 640,640
frameHeight,heightImg = 480,480
print("Import Suceesfull")
cap = cv.VideoCapture(2)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)
plateDect = cv.CascadeClassifier("Res/haarcascade_frontalface_default.xml")
count = 0



while True:
    _, img = cap.read()
    img = cv.resize(img,(frameWidth,frameHeight))

    print(img.shape)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plate = plateDect.detectMultiScale(imgGray,1.5,4)

    for x,y,w,h in plate:
        area = w*h
        
        if area>minArea:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(img,"No.Plate",(x,y-5),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0),thickness=2)
            imgPlate = img[y:y+h,x:x+w]
    
    shw("Feed",img)
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("Res/Scanned/NoPlate"+str(count)+".jpg",imgPlate)
        
        count+=1
