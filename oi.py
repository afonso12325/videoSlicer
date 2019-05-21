from kerasyolo3.yolo import YOLO
import cv2
from PIL import Image
import numpy as np


cap = cv2.VideoCapture('videos/test.mp4')
frame_count = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
fps = cap.get(cv2.CAP_PROP_FPS)
print('width = {} height = {} fps = {}'.format(width, height, fps))
out = cv2.VideoWriter('out/out_curto.avi',fourcc, 23.98, (1280,720))
while(True):
	ret, frame = cap.read()
	if ret:
            
		frame_count += 1
		#if(frame_count%100 == 0):
		print(frame_count)
		cv2.imshow('frame',frame)
		if 70<=frame_count<=73:
    			out.write(frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyAllWindows()
