import os
import cv2
import time

NUM_IMAGES = 30

if not os.path.exists('calib_data'):
  os.makedirs('calib_data')

cap = cv2.VideoCapture("tcp://192.168.1.1:5555")

for i in xrange(NUM_IMAGES):
  print '{0}/{1}'.format(i+1, NUM_IMAGES)
  img = None
  while(True):
    # flush the buffer
    for j in range(150):
      cap.read()
    ret, frame = cap.read()
    if frame != None:
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      break
    time.sleep(0.1)

  save_path = 'calib_data/img_{0}.jpg'.format(i)
  cv2.imwrite(save_path, img)
  raw_input("Saved to {0}, press Enter to continue...".format(save_path))
