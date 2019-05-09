#import pickle uncomment for python 3
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
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import os
from keras.models import load_model

def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(basic_loss)
    
    return loss


print("Loading Facial Recognition Model")
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Compiling Model ....")
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("Loading weights ....")
FRmodel.load_weights("model_weights/FRmodel.h5")
print("Loaded weights ....")
Rqconfidence=0.6
print("[INFO] loading model...Facedetection model")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
while(True):
	imageset=[]
	exists = os.path.isfile('StudentDatabase.txt')
	print("Adjust the camera and PRESS Q TO CAPTURE THE IMAGE")
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	print("starting camera")
	count=0
	while(True):	
		ret, fr = cap.read()
		cv2.imshow('frame',fr)
		frame = cv2.resize(fr,(400,400))
		(h, w) = frame.shape[:2]
		face = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
		net.setInput(face)
		detections = net.forward()
		net.setInput(face)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence < Rqconfidence:
				continue
			box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			scale=[-30, -80,30,30]
			boxfloat=box1+scale
			box=boxfloat.astype(int)
			(startX, startY, endX, endY) = box.astype("int")
			im2=frame[box[1]:box[3],box[0]:box[2]]

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			if(im2.size!=0):
				resized=cv2.resize(im2,(96,96))
				cv2.imshow("face",im2)
				count=count+1
				imageset.append(resized)
				print("count: ",count);
				if(cv2.waitKey(1) & 0xFF == ord('q')):
					cv2.destroyWindow(face)
				if (count==5):
					break
		#cv2.imshow('frame',frame)	
	imageset=np.array(imageset)
	print("Creating encodings")
	cap.release()
	cv2.destroyAllWindows()
	if exists:
		studentname=input("Enter the name: ")
		with open('StudentDatabase.txt', 'rb') as dict_items_open:
			BDICT = pickle.load(dict_items_open)
		if studentname in BDICT:
			print('Duplicate Student Name')
		else:
			BDICT[studentname] = img_to_encoding(imageset, FRmodel)
			with open('StudentDatabase.txt', 'wb') as dict_items_save:
				pickle.dump(BDICT, dict_items_save)
			print('Student successfully saved')
		
	else:
		BDICT={}
		studentname=input("Enter the name: ")
		BDICT[studentname] = img_to_encoding(imageset, FRmodel)
		with open('StudentDatabase.txt', 'wb') as dict_items_save:
			pickle.dump(BDICT, dict_items_save)
			print('Student successfully saved')
	contin=input("Enter 1 to stop new student 2 to continue ")
	if(contin=='1'):
		break
	else:
		continue
