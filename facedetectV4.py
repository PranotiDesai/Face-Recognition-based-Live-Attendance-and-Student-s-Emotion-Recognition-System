#courtsey: Andrew ng  Convolutional Neural networks couse

import pickle
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import datetime
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
calibrate=1
def facedet(frame,net):
    frame = cv2.resize(frame,(400,400))
    (h, w) = frame.shape[:2]
    faceblob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(faceblob)
    detections = net.forward()
    return detections

def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1,'%m-%d-%Y')
    d2 = datetime.datetime.strptime(d2,'%m-%d-%Y')
    return abs((d2 - d1).days)
def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(basic_loss)
    
    return loss

def findPerson(image_path, database, model):
    encoding = img_to_encodingPredict(image_path, FRmodel)
    min_dist = 300
    identity='Not known'
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc-encoding)
        if(dist < min_dist):
            min_dist = dist
            identity = name
    if(min_dist >1.5):
        identity='Not known'	
    else:
        if(calibrate==1): 
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            #identity=name
    #    identity=name    
    return min_dist, identity

Maxcount=50
Rqconfidence=0.6
print("[INFO] loading face detection model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
print("Loading Inception net Model")
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Compiling Model ....")
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("Loading weights ....")
load_weights_from_FaceNet(FRmodel)
print("Loaded weights ....")
exists = os.path.isfile('StudentDatabase.txt')
print("Loading the Data Base")
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
previous=datetime.datetime(2017, 1, 1)
previous=previous.strftime('%m-%d-%Y')
if(exists):
        with open('StudentDatabase.txt', 'rb') as dict_items_open:
            database = pickle.load(dict_items_open)
        #for key, value in database.items():
        #    print (key)
        cv2.waitKey(0) 
        while(True):
            today=datetime.datetime.now()
            today=today.strftime('%m-%d-%Y')
            daysBetw=days_between(today,previous)
            if(daysBetw>0):
                print("Creating Attendance for :"+today)
                CountAtendance={}
                TodaysAttendance={}
            ret, frame = cap.read()
            detections=facedet(frame,net)
            frame2 = cv2.resize(frame,(400,400))
            (h, w) = frame2.shape[:2]
            for i in range(0, detections.shape[2]):                                 #iterate  over the detected faces
                confidence = detections[0, 0, i, 2]                                 
                if confidence < Rqconfidence:                                       #pass the face if the detection in the blob is more than the confidence
                    continue
                box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                scale=[-30,-80,30,30]                                               #for selecting only the part of detected face
                boxfloat=box1+scale
                box=boxfloat.astype(int)
                (startX, startY, endX, endY) = box.astype("int")                    
                im2=frame2[box[1]:box[3],box[0]:box[2]]                             #grab the detected face
                if(im2.size!=0):
                    resized=cv2.resize(im2,(96,96))
                    cv2.imshow("face"+str(i),resized) 
                    detecconf,identity=findPerson(resized, database, FRmodel)      #pass the face for identity detection
                    if((identity!='Not known')):
                        if(identity in TodaysAttendance):
                            identity=identity+', Attendance Saved'
                            #print(identity+ " is Present")
							#if(CountAtendance[identity]<Maxcount):
							#	oldcount=CountAtendance[identity]
							#	newcount=oldcount+1
							#	CountAtendance[identity]=newcount
							#	identity='Present'
							#else:
								
                        else:
                            if identity in CountAtendance and CountAtendance[identity]<Maxcount:		
                                oldcount=CountAtendance[identity]
                                newcount=oldcount+1
                                if(newcount==Maxcount):
                                    TodaysAttendance[identity]=1
                                else:
                                    CountAtendance[identity]=newcount
                            else:
                                CountAtendance[identity]=1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame2, (startX, startY), (endX, endY),(0, 0, 255), 2)
                    text = identity
                    cv2.putText(frame2, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            previous=today
            cv2.imshow('frame',frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                filetosave=today+'.txt'
                if(TodaysAttendance):
                    print("Todays attendance")
                    for i in TodaysAttendance:
                        print (i)
                    f= open(filetosave,"w")                    
                    for k, v in TodaysAttendance.items():
                        f.write(str(k) + ' : '+ str(v) + '\n\n')
                    f.close()
                    print('Attendance sheet saved as:'+filetosave)
                break
else:
    print("No StudentDatabase exists")
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
