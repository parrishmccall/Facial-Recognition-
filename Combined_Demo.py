import cv2
import numpy as np
from time import sleep
from keras.models import load_model
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Model used
train_model = "ResNet"

# Size of the images
if train_model == "Inception":
    img_width, img_height = 139, 139
elif train_model == "ResNet":
    img_width, img_height = 197, 197

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
CHARACTER_FACES  = ['Andy', 'Australian', 'Conan', 'Daniel', 'Hugh']

model = load_model('ResNet-50.h5')
model2 = load_model('Conan_Resnet2.h5')


cascPath = 'haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# video_capture = cv2.VideoCapture(1)
video_capture = cv2.VideoCapture("conan.mkv")
video_capture.set(cv2.CAP_PROP_FPS, 60)

emo_list = 0
anger = 0
disgust = 0
fear = 0
happy = 0
sad = 0
surp = 0
neutral = 0

# elapsed_time = time.time() - start
# file.close()

count = 0
combined = None
top_1_prediction = None
face_pred = None

def safe_div(list, emo):
    if emo == 0:
        return 0
    else:
        return round((emo / list) * 100)


def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)

    return x

def preprocess_emotion(image):
    image = cv2.resize(image, (img_width, img_height)) 
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)

    if train_model == "Inception":
        x /= 127.5
        x -= 1.
        return x
    elif train_model == "ResNet":
        x -= 128.8006  # np.mean(train_dataset)
        x /= 64.6497  # np.std(train_dataset)

    return x


def preprocess_input2(image):
    image = cv2.resize(image, (126, 126))  
    ret = np.empty((126, 126, 1))
    ret[:, :, 0] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)
    return x

def predict(emotion):
    prediction = model.predict(emotion)

    return prediction

def predict_face(face):

    pred = model2.predict(face)

    return pred


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
    else:
        sleep(0.1)
        ret, frame = video_capture.read()  
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        image = np.zeros((20,20,3))

        faces = faceCascade.detectMultiScale(gray_frame, scaleFactor= 1.1, minNeighbors= 5, minSize= (30, 30))

        prediction = None

        prediction2 = None

        x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y: y +h, x: x +w] 

       
            cv2.rectangle(frame, (x, y), ( x +w, y+ h), (0, 0, 255), 2)

            emotion = preprocess_emotion(ROI_gray)
            face = preprocess_input(ROI_gray)

            if count % 2 == 0 and count != 0:
                prediction = predict(emotion)
                prediction2 = predict_face(face)
                print(prediction)
                #print(prediction[0][0])

                count = 0

                face_pred = CHARACTER_FACES[np.argmax(prediction2)]
                pred_str = top_1_prediction + " " + face_pred
                #print(pred_str)


            cv2.putText(frame, combined, (x, y + (h + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)
            count += 1

        frame = cv2.resize(frame, (800, 500))

        cv2.putText(img=frame, text=ratio, org=(16, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                    color=(0, 255, 0), thickness=1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(emo_list)
video_capture.release()
cv2.destroyAllWindows()
