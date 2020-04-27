import cv2
import numpy as np
from time import sleep
#from keras.models import load_model
import os

# Create a face cascade
cascPath = 'haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture("video.mkv")
video_capture.set(cv2.CAP_PROP_FPS, 30)

count = 0
frames = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
    else:
        sleep(0.0000001)
        ret, frame = video_capture.read()  # Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale

        image = np.zeros((20,20,3))

        # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
        # image:		Matrix of the type CV_8U containing an image where objects are detected
        # scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
        # minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
        # minSize:		Minimum possible object size. Objects smaller than that are ignored
        faces = faceCascade.detectMultiScale(
            gray_frame, scaleFactor	 = 1.1, minNeighbors = 5,
            minSize	= (30, 30))

        prediction = None
        x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y: y +h, x: x +w] # Extraction of the region of interest (face) from the frame

            path = 'I:/video/faces'
            # grab faces from every 3'rd frame of video
            if count % 3 == 0:
                cv2.imwrite(os.path.join(path, "secretary" + str(frames) + ".jpg"), ROI_gray)
                frames += 1
                count = 0
            count += 1

        # Display the resulting frame
        frame = cv2.resize(frame, (800, 500))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
