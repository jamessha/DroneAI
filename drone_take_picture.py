import sys
import cv2
import time

if len(sys.argv) != 2:
  raise Exception("Correct usage: python drone_take_picture <image_path>")


save_path = sys.argv[1]
cap = cv2.VideoCapture("tcp://192.168.1.1:5555")
ret, frame = cap.read()
if frame == None:
  print "Nothing in image, probably couldn't connect"
else:
  img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(save_path, img)

