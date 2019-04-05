
Project: 	Face Recognition based Attendance System

Team:

	1. Adarsh Sawant
	2. Jasmine Maria Corea 
	3. Prashant Ravindra Desai
	4. Ying Wang
	

Setup

	Following libraries needs to be installed,

		• Tensorflow
		• Keras
		• Numpy
		• Opencv
		• Pandas

Execution Steps

		• First Run AddstudentV3.py, to add the students in the system
			Take 5 pictures of student in a descent lighting condition
			Add name of student
		• Run facedetectV4.py to identify the face and mark the attendance
		• Once all students are marked, press 'Q' to save the file (Instructor's Task)
			File will get save and entitled as todays date with attended student's name.
			
References:
facenet paper:https://arxiv.org/pdf/1503.03832.pdf
Facenet model:https://github.com/davidsandberg/facenet
Inception net://https://arxiv.org/pdf/1409.4842.pdf
Face-detection using openCv/Caffe://https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
