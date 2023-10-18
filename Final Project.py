import cv2 as cv
import numpy as np
import face_recognition as fr
import os
import uuid

images = []
ids = []

# Load sample pictures and learn how to recognize it.
obama_image = fr.load_image_file("obama.jpg")
biden_image = fr.load_image_file("biden.jpeg")

images.append(obama_image)
images.append(biden_image)
ids.append("Barack Obama")
ids.append("Joe Biden")

#encode faces that has been seen before
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("encoding complete")

cap = cv.VideoCapture(0)

while True: 
    ret, frame = cap.read()

    frameS = cv.resize(frame, (0,0), None, 0.25, 0.25)
    frameS = cv.cvtColor(frameS, cv.COLOR_BGR2RGB)
    images.append(frameS)

    facesCurrFrame = fr.face_locations(frameS)
    encodesCurrFrame = fr.face_encodings(frameS, facesCurrFrame)

    #combines two args into tuples
    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        #compare unknown face to list of faces
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        id = uuid.uuid4()

        #use the known face with the smallest distance to the new face
        faceDis = fr.face_distance(encodeListKnown, encodeFace)

        matchIndex = faceDis[0]
        for dis in faceDis:
            if dis < faceDis:
                minval = dis

        # matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            id = ids[matchIndex].upper()

            y1,x2,y2, x1 = faceLoc
            y1,x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(frame, (x1,y1), (x2, y2), (255,0,255) , thickness = 2)
            cv.putText(frame, id, (x1+(x2-x1)//2,y1), cv.FONT_HERSHEY_COMPLEX,1.0, (255,255,255), thickness = 2)
        
        ids.append(id)
            
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xff==ord('d'):
        break

cap.release()
cv.destroyAllWindows
