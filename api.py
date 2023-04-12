from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
#from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
app = Flask(__name__)
#model = tf.saved_model.load("C:/Users/Subodh/OneDrive/Desktop/Hackathon/content/my_model_trained")
model = tf.keras.models.load_model("C:/Users/Subodh/OneDrive/Desktop/Hackathon/content/my_model_trained")

try:
    cap = cv2.VideoCapture(0)
except:
    print("Camera is invalid")

def process_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = preprocess_input(frame)
    prob = model.predict(np.array([frame]))
    print(prob)
    for i in prob:
        for j in i:
            #print(j)
            if (j > 0.5):
                anomaly = True
            else:
                anomaly = False
    if anomaly:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=2)
    frame = cv2.resize(frame,(512,512))
    return frame, anomaly

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, anomaly = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)