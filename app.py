import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.title("AI Emotion Detector")
st.write("Click the button below to take a photo and analyze your emotion.")

# This line automatically triggers the browser to ask for camera permission
img_file_buffer = st.camera_input("Look at the camera and smile!")

if img_file_buffer is not None:
    # Convert the file to an image that OpenCV/DeepFace can read
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner('Analyzing...'):
        try:
            # Analyze the image
            results = DeepFace.analyze(img_bgr, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
            
            # Display the result with an emoji
            if emotion == 'happy':
                st.success(f"Notification: You look Happy! 😊")
            elif emotion == 'sad':
                st.info(f"Notification: You look Sad. 😢")
            elif emotion == 'angry':
                st.error(f"Notification: You look Angry. 😠")
            else:
                st.warning(f"Notification: I detect {emotion} face.")
                
        except Exception as e:
            st.error("Could not see a face clearly. Try again!")
