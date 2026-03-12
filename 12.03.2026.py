import cv2
import os, sys, inspect

import math
import numpy as np

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# Function to return name from the dictionary
def id_class_name(class_id, classes):
    return classes[class_id]

# Find the execution path and join it with the direct reference
def execution_path(filename):
  return os.path.join(os.path.dirname(inspect.getfile(sys._getframe(1))), filename)			

# Loading model
model = cv2.dnn.readNetFromTensorflow(execution_path('models/frozen_inference_graph.pb'),
                                      execution_path('models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'))

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    image = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_height, image_width, _ = image.shape

    # Sets our input as the image, turns it into a blob
    # Resizes and sets the colour mode to BGR
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=False))

    # Returns a blob array
    output = model.forward()

    for detection in output[0, 0, :, :]:
        # detection[1] = ID, detection[2] = confidence. The other elements are the widths and heights.  

        confidence = detection[2]
        if confidence > 0.75:
            class_id = detection[1]
            class_name = id_class_name(class_id, classNames)
            print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            
            # Draw the bounding box, scaled to size of the image
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
            
            # Put some text on the bounding box
            cv2.putText(image,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))

    # cv2.resize(image, size) changes the size of an image.
    image = cv2.resize(image, (640, 480))

    # cv2.cvtColor(image, flag) transforms an image from one color space to another.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.GaussianBlur(image, kernalSize, sigmaX) applies a Gaussian function to reduce image noise and detail.
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # cv2.Canny(image, threshold1, threshold2) identifies edges.
    image = cv2.Canny(image, 30,200)

    # cv2.drawCountours(image, contours, contourIdx, color, thickness) draws outlines of detected objects.
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, contours, -1, (255,0,0),2)

    # Draw Hough Lines
    dst = cv2.Canny(image, 50, 200, None, 3)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Detected Lines (Standard Hough Transform, in red)", cdst)

    # Display
    cv2.imshow('image', image)
    cv2.imwrite("image_box_text.jpg",image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
