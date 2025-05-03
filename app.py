# Real-time emotion detection using OpenCV
from tensorflow.keras.models import load_model
from Emoxlas import preprocess_image
import time
import sys
import cv2
import numpy as np

try:
    model = load_model('model.h5')
except Exception as E:
    print("Model has not been trained")
    time.sleep(5)
    sys.exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        img_array = preprocess_image(face)
        emotion_prediction = model.predict(img_array)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()