#coding: utf-8 

import cv2 as cv 

#recuperation des fichier de ressource 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

#chargement de l'image 
img = cv.imread('obama.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#execution de la detection de visage 
faces = face_cascade.detectMultiScale(gray, 1.1, 8) #detectMultiScale(image, facteur de retrecissement, nombre de voisin)

#affichage des visages 
for face in faces:
    x, y, w, h = face 

    #rectangle autour des visages 
    cv.rectangle(img, (x, y), (x + w, y+ h), (0, 255, 0), 2)

#execution de la detection des yeux 
eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

#affichage des yeux 
for (ex, ey, ew, eh) in eyes:
    #rectangle autour des yeux 
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


#afficher l'image principale 
cv.imshow('image principal', img)
cv.waitKey(0)
cv.destroyAllWindows()