import cv2
import numpy as np
from keras.models import load_model
import firebase_admin
from firebase_admin import auth,credentials, messaging

# Load the pre-trained model
model = load_model("D:\Downloads\Main Project\detection model\Accident_detection_model.h5")

# Load the Firebase credentials
cred = credentials.Certificate('D:/Downloads/Main Project/resq-main-firebase-adminsdk-o367s-516c3dfa24.json')
firebase_admin.initialize_app(cred)

def send_notification():
    # Create a message with the required parameters
    message = messaging.Message(
        notification=messaging.Notification(
            title='Accident Detected',
            body='An accident has been detected in the video feed.'
        ),
        topic='accident_notifications'
    )
# Send message and get response
    response = messaging.send(message)
    print("Successfully sent message:", response)

# Set the threshold probability value for accident frames
threshold = 3.15


# Open the video file
cap = cv2.VideoCapture('D:\Downloads\Main Project\Datasets\Accident2.mp4')

# Get the video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create an output video writer
out = cv2.VideoWriter('D:\Downloads\Main Project\Output\output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
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
    if (predictions.max(axis=1) > threshold).any():
        label = 'Accident'
        send_notification()
    else:
        label = 'Non-accident'
        
    # Draw the label on the output frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write the output frame to the output video writer
    out.write(frame)
    
    # Display the output frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and output video writer
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
