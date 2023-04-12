import json
from flask import Flask, jsonify, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.applications.resnet import preprocess_input
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, LargeBinary, DateTime, func, String , Table, select, MetaData
from PIL import Image
import io

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

metadata = MetaData()

model = tf.keras.models.load_model("C:/Users/Subodh/OneDrive/Desktop/Hackathon/content/my_model_trained")
    
@app.route('/')
def index():
    return render_template('index.html')

def detect_anomalys(frame):

    #model = tf.keras.models.load_model("C:/Users/Subodh/OneDrive/Desktop/Hackathon/content/my_model_trained")
    
    frame = cv2.resize(frame, (64, 64))
    frame = preprocess_input(frame)
    prob = model.predict(np.array([frame]))
    #print(prob)
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
conn = engine.connect()
table = Table('Videos', metadata,schema=None)
def get_frame():
    #select_query = select(table)
    #results = conn.execute(select_query)
    results = session.query('Videos').all
    data = []
    for response in results:
        image_frame = Image.open(io.BytesIO(response.data))  #response['data']

        data.append({'id': response['id'],
                     'time': response['timestamp'],
                     'image_frame': image_frame,
                       'anomaly':response['anomaly'] })
        
    return jsonify(response)
    # return Response(json.dumps(data, default=str), mimetype='application/json')

@app.route('/video_feed')
def video():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def video_retrieve():
    return Response(get_frame())


if __name__ == '__main__':
    app.run(debug=True)

