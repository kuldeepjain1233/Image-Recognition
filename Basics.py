from cv2 import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('elon-musk-image.jpg')
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('elonmusk2.jpg')
imgtest2 = face_recognition.load_image_file('bill_gates.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
imgtest2 = cv2.cvtColor(imgtest2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgelon)[0]
# faceLoc2 = face_recognition.face_locations(imgelon)[0]
encodeElon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# cv2.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
faceLocTest2 = face_recognition.face_locations(imgtest2)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
encodeTest2 = face_recognition.face_encodings(imgtest2)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
cv2.rectangle(imgtest,(faceLocTest2[3],faceLocTest2[0]),(faceLocTest2[1],faceLocTest2[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
results2 = face_recognition.compare_faces([encodeElon], encodeTest2)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
faceDis2 = face_recognition.face_distance([encodeElon], encodeTest2)
print(results, results2, faceDis, faceDis2)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgtest2,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk', imgelon)
cv2.imshow('Elon Test', imgtest)
cv2.waitKey(0)