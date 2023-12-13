import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("keras_model.h5")

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    frame = cv2.resize(frame,(224,224))
    img = np.array(frame,dtype=np.float32)
    img = np.expand_dims(img,axis=0)
    nimg = img/255.0
    result = model.predict(nimg)
    print("Prediction: ",result)


    cv2.imshow("result",frame)
    if cv2.waitKey(25)==32:
        break
cam.release()