#coding: utf-8 

from random import choices
import cv2
import argparse
import numpy as np
from dataset import load_data 

#définir les paramètre de la ligne de commande 
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', '-c', choices=['lbp', 'eigen', 'fisher'], default='lbp')
args = parser.parse_args()

print(args.classifier)

#créer le detecteur de visages 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#charger les données de training et de test 
(X_train, y_train), (X_test, y_test) = load_data(face_cascade)

if args.classifier in ['eigen', 'fisher']:
    #redimmentionner les images
    X_train = [cv2.resize(img, (128, 128)) for img in X_train]
    X_test = [cv2.resize(img, (128, 128)) for img in X_test]

#créer l'operateur de reconnaissance faciale 
if args.classifier == 'lbp':
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
elif args.classifier == 'eigen':
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
elif args.classifier == 'fisher':
    face_recognizer = cv2.face.FisherFaceRecognizer_create()

#entrainer la reconnaissance faciale sur le set de training 
face_recognizer.train(X_train, np.array(y_train))

#Evaluer la reconnaissance faciale sur le set de test 
accuracy = 0 
for i, test_img in enumerate(X_test): 
    y_pred, confidence = face_recognizer.predict(test_img)
    if y_pred == y_pred[i]: 
        accuracy += 1 
    
accuracy = accuracy / len(X_test)
print('précision: {:.4f}'.format(accuracy))