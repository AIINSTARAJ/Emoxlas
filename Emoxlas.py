import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import time

# Load and preprocess the dataset
def load_data(data_dir):
    data = []
    labels = []
    print(f"Loading Dir {data_dir}...")
    print('Processing Images...')
    for emotion_dir in os.listdir(data_dir):
        emotion_label = emotion_dir
        emotion_folder = os.path.join(data_dir, emotion_dir)
        for image_file in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            data.append(img_to_array(resized))
            labels.append(emotion_label)
    return np.array(data), np.array(labels)

# Preprocess images for model prediction
def preprocess_image(image):
    print(f"Preprocessing {image}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Load the dataset
data_dir = r'data/train'  # Update this to the path of your dataset
data, labels = load_data(data_dir)

# Normalize and split the data
data = data.astype('float32') / 255.0
labels = LabelEncoder().fit_transform(labels)
labels = to_categorical(labels, num_classes=7)



X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=64)

model.save("model.h5")

print("Training Completed....")
time.sleep(5)