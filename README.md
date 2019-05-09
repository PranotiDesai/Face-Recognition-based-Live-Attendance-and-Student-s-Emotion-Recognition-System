# Machine-Learning-Project
## Live Attendance System and Interest Recognition 

**Team :** <br>
<br>Pranoti Desai<br>
 Rohit Diwadkar<br>
 Shailaja Yadav<br>
 Karthik Raveendran<br>
       
**Setup :** <br>
<br>Following libraries need to be installed<br>
* TensorFlow<br>
* Keras<br>
* Numpy<br>
* OpenCV<br>
* Pandas<br>
        
**Execution Steps:**<br>

* First Run the AddstudentV3.py file in order to add students to the database.<br>
    As the camera pops up, take 5pictures of student in a descent lighting condition.v
    Save the name of the student.<br>
* Follow the instructions to add the students. Press 1 to stop, 2 to continue adding students.<br>
* A file Studentdatabase.txt will be created with the encodings of the students.<br>

* Run the facedetectV4.py file to identify faces and emotions of the students. <br>
* The emotion recognition model is saved in a h5 file.<br>
* The pipeline model loads the saved model and gives the emotions of the students.<br>

* Press Q to quit

**Results : **
![Picture1](https://user-images.githubusercontent.com/43567199/57425553-9608f980-71e9-11e9-97c2-6191b6f9e398.png)









