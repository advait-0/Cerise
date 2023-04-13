from flask import Flask,render_template,Response
import cv2
import os
import secrets
from PIL import Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import time
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
import MySQLdb.cursors
import mysql.connector
import datetime


app=Flask(__name__)

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'pass1234'
# app.config['MYSQL_DB'] = 'anomaly_detect'
 
# mysql = MySQL(app)

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

# def save_picture(form_picture):
#     random_hex = secrets.token_hex(8)
#     _, f_ext = os.path.splitext(form_picture.filename)
#     picture_fn = random_hex + f_ext
#     picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

#     i = Image.open(form_picture)
#     i.save(picture_path)

#     return picture_fn

# model = keras.models.load_model("my_model_trained/content/my_model_trained")

def add_data(path, anomaly):
     try:
        connection = mysql.connector.connect(host='localhost',
                                            database='anomaly_detect',
                                            user='root',
                                            password='pass1234')

        mySql_insert_query = """INSERT INTO video (img_path, anomaly) 
                            VALUES 
                            (%s , %s) """
        
        record = (path, anomaly)

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query, record)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Laptop table")
        cursor.close()

     except mysql.connector.Error as error:
        print("Failed to insert record into Laptop table {}".format(error))

     connection.close()


def generate_frames():
    success = 1
    # time_now = time.time() 
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
                    count += 1
                    

        else:
            print("Anomaly")
            parent_dir = "C:/Users/MIHIR RATHOD/Desktop/Kavach"
            child_dir = "/anomalies/"
            current_time = datetime.datetime.now().strftime('%d-%m-%y-%H-%M-%S')
            filename = current_time + '.png'
            #print(current_time)

            path = parent_dir + child_dir + filename

            directory = r'C:\\Users\\MIHIR RATHOD\Desktop\\Kavach\\anomalies'

            os.chdir(directory) 

            # Print the list of files in the directory before saving the image
            # print("Before saving")   
            # print(os.listdir(directory))   

            # Save the image with the filename "cat.jpg"
            
            cv2.imwrite(filename, frame) 
            # frame.save(f'{path}')

            
            add_data(path, "Anomaly")
            # with app.app_context():
            #     cur = mysql.connection.cursor()
            # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            # cursor.execute('INSERT INTO video (img_path, anomaly) VALUES (% s, % s)', (path, "anomaly"))
         
            count = 0
          
            

        if not success:
            break
        else:        

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        # model.predict(frame)
        # print("FPS: {0}".format(int(fps)))
        # print(prob)
        
        time.sleep(0.033)
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

conn = MySQLdb.connect("localhost","root","pass1234","anomaly_detect" ) 
cursor = conn.cursor()

@app.route('/display.html')
def display_db():
    #  cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    #  cursor.execute('SELECT * FROM video')
    cursor.execute("SELECT * FROM video") 
    data = cursor.fetchall() #data from database 
    return render_template("display.html", value=data)     
     

if __name__=="__main__":
    app.run(debug=True)


