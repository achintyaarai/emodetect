import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import deque
import matplotlib.pyplot as plt

# Load the trained model
model_best = load_model('face_model.h5')  # Set your model file path here

# Classes for 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Predefined colors for each emotion
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgusted': (0, 255, 0),  # Green
    'Fear': (0, 0, 10),        # Blue
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 255, 0),      # Cyan
    'Surprise': (0, 165, 255), # Orange
    'Neutral': (128, 128, 128) # Gray
}

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Initialize variables to store last predictions
emotion_history = deque(maxlen=20)  # Store last 20 emotions for history display
color_history = deque(maxlen=100)    # Store the average color history for the graph

# Create a matplotlib figure for the color graph
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
line, = ax.plot([], [], color='black')  # Initialize a line object
ax.set_xlim(0, 100)  # X-axis limits
ax.set_ylim(0, 255)  # Y-axis limits for RGB
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Color Intensity')
ax.set_title('Average Mood Color Over Time')
plt.show()

# Main loop for emotion detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there's an issue with the camera

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White rectangle

        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)

        # Predict emotion using the loaded model
        predictions = model_best.predict(face_image)

        # Get the index of the highest prediction
        emotion_index = np.argmax(predictions)
        detected_emotion = class_names[emotion_index]

        # Add detected emotion to the history
        emotion_history.append(detected_emotion)

        # Calculate the average mood color from the last 20 emotions
        avg_color = np.mean([emotion_colors[emotion] for emotion in emotion_history], axis=0)
        avg_color = tuple(map(int, avg_color))  # Convert to integer RGB values

        # Draw a rectangle at the top of the frame with the average mood color
        cv2.rectangle(frame, (0, 0), (640, 50), avg_color, -1)

        # Display the detected emotion on the frame
        cv2.putText(frame, f"Emotion: {detected_emotion}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Store the average color in the color history for graphing
        color_history.append(avg_color)

    # Update the graph
    if len(color_history) > 0:
        r_values = [c[0] for c in color_history]
        g_values = [c[1] for c in color_history]
        b_values = [c[2] for c in color_history]
        
        ax.clear()  # Clear the previous graph
        ax.plot(b_values, color='red', label='Red Channel')
        ax.plot(g_values, color='green', label='Green Channel')
        ax.plot(r_values, color='blue', label='Blue Channel')
        ax.legend()
        ax.set_xlim(0, 100)  # X-axis limits
        ax.set_ylim(0, 255)  # Y-axis limits for RGB
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Color Intensity')
        ax.set_title('Average Mood Color Over Time')
        plt.pause(0.001)  # Pause to allow the graph to update

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
plt.close(fig)  # Close the matplotlib figure
