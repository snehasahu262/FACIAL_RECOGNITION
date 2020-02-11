import cv2
import numpy as np
import os
import sys
import errno
import time
import re

import glob
import shutil

user_name=raw_input("enter user name")
print(user_name)





def create_dir():
    if not os.path.exists("./training-images/"+user_name):
        try:
            os.makedirs("./training-images/"+user_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
   
    
create_dir()

	
	
	
	
	
current_path="./training-images/"
filename=user_name+"/"
donepath=os.path.join(current_path,filename)

print(donepath)





# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    #for (x,y,w,h) in faces:
        #cropped_face = img[y:y+h, x:x+w]

    return faces

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 20 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        #face = cv2.resize(face_extractor(frame), (1280, 720))
		
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
		
		#file_name_path = donepath + str(count) + '.jpg'
        file_name_path = donepath+ str(count) + '.jpg'
        cv2.imwrite(file_name_path, frame)

        # Put count on images and display live count
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', frame)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 20: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
