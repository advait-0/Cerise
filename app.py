from flask import Flask,render_template,Response
import cv2
import os
import secrets
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import time
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, LargeBinary, DateTime, func, String , Table, select, MetaData
from PIL import Image
import io

app=Flask(__name__)

engine = create_engine('mysql://root:pass1234@localhost/anomaly_detect')
Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    img_path = Column(String(70))
    anomaly = Column(String(70))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

metadata = MetaData()

preprocess_fun = tf.keras.applications.densenet.preprocess_input
datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function = preprocess_fun)

model = keras.models.load_model("my_model_trained/content/my_model_trained")

def detect_anomaly(image):
    image = cv2.resize(image, (64,64))
    img_tensor = np.expand_dims(image, axis=0)
    #Creates our batch of one image
    pic = datagen.flow(img_tensor, batch_size =1)
    predictions = model.predict(pic)
    return predictions[0][7]

camera= cv2.VideoCapture("C:/Users/MIHIR RATHOD/Downloads/test_cute_cat.mp4")
# cv2.VideoCapture(0)

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    i = Image.open(form_picture)
    i.save(picture_path)

    return picture_fn

# model = keras.models.load_model("my_model_trained/content/my_model_trained")


def generate_frames():
    success = 1
    time_now = time.time() + 60 
    count = 0
    while success:
            
        ## read the camera frame
        success,frame=camera.read()
    
        # res_frame = cv2.resize(frame, (64,64))

        # images_list = []
        # images_list.append(np.array(res_frame))
         
        # x = np.asarray(images_list)
        # # x = cv2.resize(x, (64,64))
        if count < 10:

            if detect_anomaly(frame) > 0.5:
                if time.time() - time_now > 60:
                    count += 1

        else:
            time_now = time.time()
            count = 0
            parent_dir = app.root_path
            child_dir = "/anomalies"

            path = os.path.join(parent_dir, child_dir)
            i = Image.open(frame)
            i.save(path)
            
            vid = Video()  
            vid.anomaly = "Anomaly"         
            vid.img_path = path

        if not success:
            break
        else:        

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        # model.predict(frame)
        # print("FPS: {0}".format(int(fps)))
        # print(prob)
        

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)


