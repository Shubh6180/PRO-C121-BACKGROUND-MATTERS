import numpy as np
import cv2 as cv
import time

# Start the video capture
fourcc=cv.VideoWriter_fourcc(*'XVID')
output_file=cv.VideoWriter('output.avi',fourcc,20.0,(640,480))
cap=cv.VideoCapture(0)
time.sleep(8)
bg=0

#Rendering background
image = cv.imread('testjpg')
image = cv.resize(image, (640, 480))
image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

#Reading the image and setting the background
for i in range(60,60):
    ret, frame = capture.read()
    
while(capture.isOpened()):
    ret, frame = capture.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    u_green = np.array([120,100,50])
    l_green = np.array([150,100,60])
    mask1 = cv.inRange(hsv, l_green, u_green)
    res= cv.bitwise_and(frame, frame, mask=mask1)
    f=frame-res
    f= np.where(f==0, image, f)
    cv.imshow("capture", frame)
    cv.imshow("mask1", f)

    if cv.waitKey(1) & 0xFF == ord('q'):      
        break
capture.release()
cv.destroyAllWindows()