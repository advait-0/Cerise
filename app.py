from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.applications.resnet import preprocess_input
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, LargeBinary, DateTime, func, String

app = Flask(__name__)
engine = create_engine('mysql://root:subumitu#1@localhost/dbname')
Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    data = Column(LargeBinary)
    anomaly = Column(String(70))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/')
def index():
    return 'Hello World'

def detect_anomalys(frame):

    model = tf.keras.models.load_model("C:/Users/Subodh/OneDrive/Desktop/Hackathon/content/my_model_trained")
    
    frame = cv2.resize(frame, (64, 64))
    frame = preprocess_input(frame)
    prob = model.predict(np.array([frame]))
    print(prob)
    for i in prob[0]:
        #print(i)
        if (prob[0][0] > 0.5):
            anomaly = 'Crime1'
        elif(prob[0][1] > 0.5):
            anomaly = 'Crime2'
        elif(prob[0][2] > 0.5):
            anomaly = 'Crime3'
        elif(prob[0][3] > 0.5):
            anomaly = 'Crime4'
        elif(prob[0][4] > 0.5):
            anomaly = 'Crime5'
        elif(prob[0][5] > 0.5):
            anomaly = 'Crime6'
        elif(prob[0][6] > 0.5):
            anomaly = 'Crime7'
        elif(prob[0][7] > 0.5):
            anomaly = False
        elif(prob[0][8] > 0.5):
            anomaly = 'Crime8'
        elif(prob[0][9] > 0.5):
            anomaly = 'Crime9'
        elif(prob[0][10] > 0.5):
            anomaly = 'Crime10'
        elif(prob[0][11] > 0.5):
            anomaly = 'Crime11'
        elif(prob[0][12] > 0.5):
            anomaly = 'Crime12'
        elif(prob[0][13] > 0.5):
            anomaly = 'Crime13'
        else:
            anomaly = False
    if anomaly != False:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=2)
    frame = cv2.resize(frame,(64,64))
    return frame , anomaly

cap = cv2.VideoCapture(0)

'''def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, anomaly = detect_anomalys(frame)

        #print(processed_frame)
    
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')'''

def process_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        pred,anomaly = detect_anomalys(frame)
        
        # Store the first frame that detects a model probability higher than 0.5

        img = Video(data=pred.tobytes(),anomaly=anomaly )
        session.add(img)
        session.commit()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()

'''def gen():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        

        frame = detect_anomalys(frame)

        with Session() as session:
            img = Video(data=frame1)
            session.add(video)
            session.commit()

            img = Video(frame=frame.tobytes())
            session.add(img)
            session.commit()

            
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()'''

@app.route('/video_feed')
def video():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

'''@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
'''
if __name__ == '__main__':
    app.run(debug=True)

