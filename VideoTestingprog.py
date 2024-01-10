from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
import math

cap=cv2.VideoCapture(r'videotest.mp4')
model=YOLO(r'fin_model.pt')


classNames=['Glacier', 'Grassland', 'Mountain', 'Plain']
myColor=(0,0,255)


while(True):
      
    ret, frame = cap.read()
    results=model(frame, stream=True)
    for r in results:
        boxes=r.boxes
        print("Value of Boxes:",boxes)
        for box in boxes:
                print("Value of Box:",box)
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1,  y2-y1
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                currentClass= classNames[cls]
                print(currentClass)
                cv2.rectangle(frame,(x1,y1),(x2,y2),myColor,2)
                cv2.putText(frame, currentClass, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, myColor, 2)
  
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()


