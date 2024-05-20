import cv2
from PIL import Image #bounding box for the item
from util import get_limits #custom .py file
from collections import deque

yellow = [0,255,255] #yellow in BGR colorspace
cap = cv2.VideoCapture(0)
lower_fps_limit = 10  # Minimum FPS
upper_fps_limit = 30  # Maximum FPS
bbox_queue = deque(maxlen=10) #deque to reduce jitteriness of the bonding box

while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerlimit, upperlimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage,lowerlimit, upperlimit) # get a mask of the pixels we want to be in the image
    mask_ = Image.fromarray(mask) #convert the numpy array into pillow
    bbox = mask_.getbbox() #get a bounding mask
    if bbox is not None:
        bbox_queue.append(bbox)
        avg_bbox = [sum(coords) // len(bbox_queue) for coords in zip(*bbox_queue)]
        x1, y1, x2, y2 = avg_bbox
        cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),5 ) #upper left corner, right bottom corner
    cv2.imshow('frame',frame)
    delay = max(1, int(1000 / upper_fps_limit))  # Delay in milliseconds
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    
    if delay < (1000 / lower_fps_limit):
        continue
    
cap.release()

cv2.destroyAllWindows()
    
