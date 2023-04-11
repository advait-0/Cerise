from flask import Flask,render_template,Response
import cv2
import tensorflow as tf
from tensorflow import keras
import time

app=Flask(__name__)
camera=cv2.VideoCapture(0)

model = keras.models.load_model("my_model_trained/content/my_model_trained")

def generate_frames():
    while True:
            
        ## read the camera frame
        prev = time.time()
        success,frame=camera.read()
        curr = time.time()

        if not success:
            break
        else:        

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        fps = 1/(curr - prev)
        # model.predict(frame)
        # print("FPS: {0}".format(int(fps)))
        

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