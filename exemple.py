import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# 📂 Dossier contenant les visages à reconnaître
path = '/Users/salim/Desktop/projet_pointage/faces'  
images = []
classNames = []
myList = os.listdir(path)

# Chargement des images
for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

# Encodage des visages connus
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = findEncodings(images)
print('✅ Visages encodés !')

# Fonction d'enregistrement dans un fichier CSV
def markAttendance(name):
    df_file = 'presence.csv'
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(df_file):
        df = pd.read_csv(df_file)
    else:
        df = pd.DataFrame(columns=['Nom', 'Heure'])

    if name not in df['Nom'].values:
        df.loc[len(df)] = [name, dt_string]
        df.to_csv(df_file, index=False)
        print(f'✔️ Présence enregistrée pour {name} à {dt_string}')

# Démarrer la webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # + rapide
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam - Pointage Automatique', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
        break

cap.release()
cv2.destroyAllWindows()
