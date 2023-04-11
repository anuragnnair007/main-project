import os
import cv2
import numpy as np
from tensorflow import keras

# Define parameters for image processing and classification
IMG_HEIGHT = 224
IMG_WIDTH = 224
THRESHOLD = 1.54

# Load the trained MobileNetV2 model for image classification
model = keras.models.load_model('D:\Downloads\Main Project\detection model\Accident_detection_model.h5')

# Load the video file
video_file = 'D:\Downloads\Main Project\Datasets\Accident2.mp4'
cap = cv2.VideoCapture(video_file)

# Create a new directory to save the classified frames


# Loop through the frames of the video and classify each one
frame_num = 0
while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the input size of the model and normalize pixel values
    resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    normalized_frame = resized_frame / 255.0

    # Classify the frame using the pre-trained model
    prediction = model.predict(np.expand_dims(normalized_frame, axis=0))[0]

    # Determine whether the frame is an accident or non-accident image based on the predicted probability
    if prediction[1] > THRESHOLD:
        label = 'accident'
        output_folder = 'D:\Downloads\Main Project\Frames\Accident'
        os.makedirs(output_folder, exist_ok=True)
    else:
        label = 'non-accident'
        output_folder = 'D:\Downloads\Main Project\Frames\Accident_not'
        os.makedirs(output_folder, exist_ok=True)

    # Save the classified frame to the output folder
    output_file = os.path.join(output_folder, f'frame_{frame_num:04d}_{label}.jpg')
    cv2.imwrite(output_file, frame)

    # Print progress information
    print(f'Processed frame {frame_num} - predicted class: {label}')

    # Increment the frame counter
    frame_num += 1

# Release the video capture and cleanup
cap.release()
cv2.destroyAllWindows()
