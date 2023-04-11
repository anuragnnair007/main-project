import cv2
import os

# Define the path to the video file
video_path = 'D:\Downloads\Main Project\Datasets\Accident.mp4'

# Define the path to the directory where the frames will be saved
frames_path = 'D:\Downloads\Main Project'

# Create the frames directory if it doesn't exist
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

# Load the video file
cap = cv2.VideoCapture(video_path)

# Initialize frame counter
frame_num = 0

# Loop through each frame in the video
while cap.isOpened():

    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:

        # Define the filename for the current frame
        filename = os.path.join(frames_path, f'frame_{frame_num:05d}.jpg')

        # Save the current frame as an image file
        cv2.imwrite(filename, frame)

        # Increment the frame counter
        frame_num += 1

    # If the frame was not successfully read, break out of the loop
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
