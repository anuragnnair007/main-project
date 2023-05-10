import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Set up the parameters
batch_size = 32
img_height = 224
img_width = 224
num_classes = 2

# Load the dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
  "D:\Downloads\Main Project\Frames",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(
  'D:\Downloads\Main Project\Val',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\Downloads\Main Project\Frames',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)

# Create the model
model = keras.Sequential([
tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
layers.Conv2D(32, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(128, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(64, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_classes)
])

# Compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Train the model
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

# Save the model
model.save('D:\Downloads\Main Project\detection model\Accident_detection_model.h5')
