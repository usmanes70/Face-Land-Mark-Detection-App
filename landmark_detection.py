#import libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
import urllib.request as urlreq
import os

# Put Title
st.title('Face Land Mark Detection App')
st.header('Introduction')
st.markdown('''
Face landmark detection is a computer vision task where we want to detect and track keypoints from a human face.
Facial landmark detection is the process of detecting landmarks or regions of interest (key-points) 
on the face like Eyebrows, Eyes, Nose,Mouth and Jaw silhouette.
Some applications of facial landmark detection are face swap, head pose detection, 
detecting facial gestures, gaze direction etc.''')

st.header('Classifiers Used')
st.markdown('''In this app, two classifiers are used. One is **haarcascade frontal face** classifier
which is used for face detection and for detecting face landmarks use **LBF model**''')

# Upload Image
st.sidebar.header("Upload an Image File")
st.sidebar.markdown('**Upload the image of your choice from your local storage**')
upload_image = st.sidebar.file_uploader("Upload image", type=["JPEG", "JPG", "PNG"])

# Read an image
if upload_image is not None:
    image = Image.open(upload_image)
    image = np.array(image)
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier()

    # #-----------------------------------------------------------
    # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        print("File exists")
    else:
     # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")
    
    # #-------------------------------------------------------------------
    #Create cascade classifier
    face_cascade = cv.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    # Draw a rectangle

    for (x, y, w, h) in faces:
    
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    #---------------------------------------------------------------------
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "lbfmodel.yaml"

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
         print("File exists")
    else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")

    #-----------------------------------------------------------------------
    # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(gray_img, faces)

    for landmark in landmarks:
        for x,y in landmark[0]:
		    # display landmarks on "image_cropped"
		    # with white colour in BGR and thickness 1
            cv.circle(image, (int(x), int(y)), 1, (181, 24, 115), 2)

    #Show face land mark image
    st.markdown("**Face Land Mark Points**")
    st.image(image)


