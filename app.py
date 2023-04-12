from flask import Flask,render_template,Response
import cv2
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app=Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
camera=cv2.VideoCapture(0)

model = keras.models.load_model("my_model_trained/content/my_model_trained")

def generate_frames():
    while True:
            
        ## read the camera frame
        prev = time.time()
        success,frame=camera.read()
        curr = time.time()
        res_frame = cv2.resize(frame, (64,64))

        images_list = []
        images_list.append(np.array(res_frame))
        x = np.asarray(images_list)
        # x = cv2.resize(x, (64,64))
        prob = model.predict(x)
        
        
        if not success:
            break
        else:        

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        fps = 1/(curr - prev)
        # model.predict(frame)
        # print("FPS: {0}".format(int(fps)))
        
        print(prob)

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