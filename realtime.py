#coding: utf-8 

import cv2 
import numpy as np 

from dataset import load_data 

#detecteur de visages 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
(X_train, y_train), (X_test, y_test) = load_data(face_cascade)

#operateur de reconaissanse faciale 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(X_train, np.array(y_train))

#open webcam 
video_capture = cv2.VideoCapture(0)
while True:
    #obtenir image depuis la webcam 
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detection de visage sur image
    faces = face_cascade.detectMultiScale(gray)
    for face in faces:
        #extraction du visage 
        x, y, w, h = face 
        face_region = gray[y:y+h,x:x+w]

        #executer la reconnaissance faciale
        predicted_person, _ = face_recognizer.predict(face_region)

        #rectangle sur visage 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(predicted_person), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    #afficher l'image annotée
    cv2.imshow('Reconnaissance faciale', frame)

    #quitter l'application 
    if cv2.waitKey(1) == ord('q'):
        break
#fin de la boucle 

#close webcam
video_capture.release()
cv2.destroyAllWindows()