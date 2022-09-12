ROBOFLOW_API_KEY = "FW7IgInblNz6JrOTeE3k"
ROBOFLOW_MODEL = "FILL YOUR MODEL ID HERE" # eg xx-xxxx--#
ROBOFLOW_SIZE = 416
import cv2 as cv
import numpy as np

#LOAD YOLO
Net =   cv.dnn.readNet()

#extract object names from coco and kept it in list
classes = []
#with open('rugs_data.yaml','r') as f:
    #classes = f.read().splitlines() #the splitlines() method splits a string into a list.
#for webcam/video
cap = cv.VideoCapture("carpets.mp4")

while True:
     ret,img = cap.read()
     height,width,channels = img.shape

     blob = cv.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
     Net.setInput(blob)
     output_layers_names = Net.getUnconnectedOutLayersNames()
     layerOutputs = Net.forward(output_layers_names)
      #initializing lists
     boxes=[]
     confidences=[]
     class_ids=[]

     for output in layerOutputs:
        for detection in output:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence>0.5:
               #YOLO predicts the result with the center of the bounding box.
                  center_x = int(detection[0]*width)
                  center_y = int(detection[1]*height)
                  w = int(detection[2]*width)
                  h = int(detection[3]*height)

                  x = int(center_x - w/2)
                  y = int(center_y - h/2)

                  boxes.append([x,y,w,h])
                  confidences.append(float(confidence))
                  class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
        font = cv.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size=(len(boxes),3))

        #loop for each object detected and extract info from boxes.
        if len(indexes) > 0: # You have to check for 0 length indexes before the for loop
         for i in indexes.flatten():
          x,y,w,h = boxes[i]
          label = str(classes[class_ids[i]])
          confidence = str(round(confidences[i],2))
          color = colors[i]
          cv.rectangle(img,(x,y),(x+w,y+h),color,2)
          cv.putText(img,label+""+confidence,(x,y+20),font,1,(255,0,0),2)

     cv.imshow('Image',img)
     if cv.waitKey(5) & 0xFF == ord('q'):
      break

cap.release()
cv.destroyAllWindows()
    