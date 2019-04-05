# Face-Recognition-based-Attendance-System
Face recognition based attendance system using Facenet


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
      Follow instruction to add the student .(Press 1 or 2 according to your needs)
      A file Student database.txt will be created with the encodings of images.
		• Run facedetectV4.py to identify the face and mark the attendance
		• Once all students are marked, press 'Q' to save the file (Instructor's Task)
			File will get save and entitled as todays date with attended student's name.ex today us 12-5-2018 the file will be saved as 12-5-2018.txt
