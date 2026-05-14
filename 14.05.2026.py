import cv2
import serial

import statistics
import numpy as np
import time

arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(3)
arduino.reset_input_buffer()
print("Serial OK")

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

while True:
    width = 640
    height = 480

    check, frame = cam.read()
    image = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_height, image_width, _ = image.shape

    # Draw Hough Lines
    dst = cv2.Canny(image, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    leftPoints = []
    rightPoints = []
    imageCentreX = width // 2   
    
    # Probabilstic Hough Lines
    linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 50, None, 100, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]

            # Draw lines
            cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)  

            # Gradient
            dx = x2 - x1
            dy = y2 - y2	

            if dx != 0: 
                m = round(((y2 - y1) / (x2 - x1)), 2)

            # Midpoints
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            #cv2.circle(cdstP, (x1, y1), 5, (255, 0, 0), 3)
            #cv2.circle(cdstP, (x2, y2), 5, (0, 255, 0), 3)
            cv2.circle(cdstP, (cx, cy), 5, (255, 255, 255), 3)

            # Sorting
            if cx < imageCentreX:
                leftPoints.append(cx)
            else:
                rightPoints.append(cx)

    if (len(leftPoints) != 0) and (len(rightPoints) != 0):
        leftTapeX = statistics.mean(leftPoints)
        rightTapeX = statistics.mean(rightPoints)
        
        laneCentreX = (leftTapeX + rightTapeX) / 2
        error = round((laneCentreX - imageCentreX) / 10, 2) 
        cv2.putText(cdstP, str(error), (width // 2 + 50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        
        #print("Sending message to Arduino")
        arduino.write((str(error) + "\n").encode('utf-8'))
    
    cv2.line(cdstP, (width // 2, height), (width // 2, 0), (0, 255, 255), 2, cv2.LINE_AA)  
    
    #cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
arduino.close() 
