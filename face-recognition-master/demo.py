import cv2

cap=cv2.VideoCapture(0)
ret,frame=cap.read()

while(True):
    cv2.imshow('img1',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('abc.jpg',frame)
        cv2.destroyAllWindows()
        break

cap.release()

