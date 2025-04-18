#coding: utf-8 

import cv2 as cv 
import sys

#recuperation du classificateur en cascade pre-entrainer
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#chargement de l'image 
img = cv.imread('brad-angelina.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#execution de la detection de visage 
faces = face_cascade.detectMultiScale(gray, 1.1, 8) #detectMultiScale(image, facteur de retrecissement, nombre de voisin)

#verifier le nombre de visage 
if len(faces) != 2:
    sys.exit('la photo doit avoir exactement 2 visages, réessayer...')

# recuperation des dim de chaque visage 
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

#extraction des visages
face1 = img[y1:y1+h1,x1:x1+w1] 
face2 = img[y2:y2+h2,x2:x2+w2]

#redimensionnement des visages 
face2 = cv.resize(face2,(w1,h1))
face1 = cv.resize(face1,(w2,h2))

#remplacement de visages
img[y2:y2+h2,x2:x2+w2] = face1

img[y1:y1+h1,x1:x1+w1] = face2

#affichage de l'echange des visages
cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()

