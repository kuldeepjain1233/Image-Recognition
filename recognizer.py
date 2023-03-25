from cv2 import cv2
import numpy as np
import face_recognition
import os

path = 'total_images'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0),None,1.04,1.04)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facescurframe = face_recognition.face_locations(imgs)
    encodescurframe = face_recognition.face_encodings(img, facescurframe)

    for encodeface, faceLoc in zip(encodescurframe, facescurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        faceDis = face_recognition.face_distance(encodelistknown, encodeface)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),int(0.7))

    cv2.imshow('webcam', img)
    cv2.waitKey(1)


# faceLock = face_recognition.face_locations(imgelon)[0]
# encodeElon = face_recognition.face_encodings(imgelon)[0]
# cv2.rectangle(imgelon,(faceLock[3],faceLock[0]),(faceLock[1],faceLock[2]),(255,0,255),2)

# faceLockTest = face_recognition.face_locations(imgelon)[0]
# encodeTest = face_recognition.face_encodings(imgtest)[0]
# cv2.rectangle(imgtest,(faceLockTest[3],faceLockTest[0]),(faceLockTest[1],faceLockTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon], encodeTest)
# faceDis = face_recognition.face_distance([encodeElon], encodeTest)