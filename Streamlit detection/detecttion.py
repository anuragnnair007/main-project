import cv2
import numpy as np
import streamlit as st
import os
import tensorflow
import keras

# Load the pre-trained model
model = keras.models.load_model('D:\Downloads\Main Project\detection model\Accident_detection_model.h5')

# Function to preprocess a frame
def preprocess_frame(frame):
    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Normalize pixel values to the range [0, 1]
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to match the model's input shape
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)

    return preprocessed_frame

# Function to perform accident detection on a preprocessed frame
def detect_accident(frame):
    # Perform accident detection using the pre-trained model
    prediction = model.predict(frame)[0]

    # Determine the predicted class based on the probability
    if prediction > 3.15:

        return 'Accident'

    else:
        return 'Non-Accident'

def main():
    st.title("Accident Detection System")

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = "./temp.mp4"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.read())

        # Read the video using OpenCV
        video = cv2.VideoCapture(temp_file)

        # Check if the video was opened successfully
        if not video.isOpened():
            st.error("Failed to open the video file.")
        else:
            # Read and display the video frames
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Pre-process the input frame
                img = cv2.resize(frame, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict the probability of the input frame being an accident frame
                predictions = model.predict(img)
                
                # Classify the frame as an accident or non-accident frame based on the threshold probability
                if (predictions.max(axis=1) > 3.15).any():
                    label = 'Accident'
                else:
                    label = 'Non-accident'
                    
                # Draw the label on the output frame
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Write the output frame to the output video writer
                #st.write(label)
                
                # Display the output frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


                # Display the frame
                #st.image(frame, channels="BGR")

        # Release the video capture object and delete the temporary file
        video.release()
        os.remove(temp_file)

    else:
        st.warning("Please upload a video file.")

if __name__ == "__main__":
    main()

