"""""""""""""""""""""""""""""""""""""""""""""
Project: Object Detection
Description: object detector which identifies the classes of the objects in 
             an image or video.
By:- Shashank GS
(Project implemented during my internship period in The Sparks Foundation)
(Task 1)
"""""""""""""""""""""""""""""""""""""""""""""
#importing the required libraries
import numpy as np
import cv2

thres = 0.5 # Threshold to detect object
nms_threshold = 0.1 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture('./InputVideos/cycle.mp4')

#extracting the names of differnt objects/classes
#from the "coco.names" file into classNames list
classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()

#setting the random color array
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

"""
I am going to use mobilenet ssd 
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) 
network intended to perform object detection.
"""
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

"""
The algorithm to detect the object in the video and draw a rectangle
around the region of interet and labeling it with the text
"""
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    if len(classIds) != 0:
        for i in indices:
            i = i[0]
            box = bbox[i]
            confidence = str(round(confs[i],2))
            color = Colors[classIds[i][0]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i][0]-1]+" "+confidence,(x+10,y+20),
                        cv2.FONT_HERSHEY_PLAIN,1,color,2)
    cv2.imshow("Output",img)
    k=cv2.waitKey(10)
    if k==27:
        break
    #press escape 'Esc' to exit the window
#ending
cap.release()
cv2.destroyAllWindows()
