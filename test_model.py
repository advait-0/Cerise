import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np

preprocess_fun = tf.keras.applications.densenet.preprocess_input
datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function = preprocess_fun)

def detect_anomaly(image):
    image = cv2.resize(image, (64,64))
    img_tensor = np.expand_dims(image, axis=0)
    #Creates our batch of one image
    pic = datagen.flow(img_tensor, batch_size =1)
    predictions = model.predict(pic)
    return predictions[0][7]

model = keras.models.load_model("my_model_trained/content/my_model_trained")

camera= cv2.VideoCapture("C:/Users/MIHIR RATHOD/Downloads/test_cute_cat.mp4")

# checks whether frames were extracted
success = 1
  
while success:
  
    # vidObj object calls read
    # func tion extract frames
    success, image = camera.read()
  
    # Saves the frames with frame-count
    if(detect_anomaly(image) > 0.5):
        print("Normal")
    else:
        print("Anomaly!")