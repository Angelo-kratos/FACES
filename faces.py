import numpy as np
import mediapipe as mp
import cv2 as cv
import face_recognition
import time

face_cascade = cv.CascadeClassifier("/home/kratos/PycharmProjects/pythonProject/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
picture = cv.VideoCapture(0)
opened = picture.isOpened()
end = 0
if opened:
    while True:
        _,frame = picture.read()
        #frame = cv.resize(frame,(400,400),interpolation=cv.INTER_CUBIC)

        start = time.time()
        timeTotal = 1 // (start-end)
        end = start

        #Detecter les points pour tracer le triangle
        grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)


        for (x,y,w,h) in faces:   ##Tracer le triangle sur la partie du visage
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            img_croped = frame[y-100:y+h+100,x-100:x+w+100]   #couper la partie du visage
            cv.imshow("img", img_croped)
            cv.imwrite("Cap3.jpg",img_croped)


        image1 = face_recognition.load_image_file("/home/kratos/Documents/opencv/andry.jpg")  #load file vient du capturer( file model)
       # Encodage et detection de la partie du visage
        face_loc1 = face_recognition.face_locations(image1)[0]
        face_encoded1 = face_recognition.face_encodings(image1)[0]

        image2 = face_recognition.load_image_file("/home/kratos/Documents/opencv/andry1.jpeg") #load file capturel en temps reel
       # Encodage et detection de la partie du visage
        face_loc2  = face_recognition.face_locations(image2)[0]
        face_encoded2 = face_recognition.face_encodings(image2)[0]
        
        
        result =face_recognition.compare_faces([face_encoded1],face_encoded2)
        face_dist = face_recognition.face_distance([face_encoded1],face_encoded2)
        print(result,face_dist)


        cv.putText(frame,str(f'FPS:{timeTotal}'),(5,15),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)#Afficher FPS
        cv.imshow("Videos", frame)
        keystore = cv.waitKey(1)
        if keystore == 27:
            break

picture.release()
cv.destroyAllWindows()
