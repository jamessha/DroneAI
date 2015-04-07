import os
import libardrone
import time
import cv2
import pygame
import numpy as np

from ar_markers.hamming.detect import detect_markers
#from collections import Queue
from multiprocessing import Queue, Process

MARKER_SIZE = 18.7 # In cm

BASE_MARKER = np.array([[1., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0.],
                        [1., 0., 0., 1., 0.],
                        [1., 1., 0., 0., 0.],
                        [0., 0., 1., 1., 0.]])

def parse_calib(filepath):
  f = open(filepath, 'r')
  K = np.zeros((3, 3), dtype=np.float32)
  d = np.zeros((1, 5), dtype=np.float32)
  for i in xrange(3):
    toks = f.readline().split()
    for j in xrange(3):
      K[i, j] = float(toks[j])

  f.readline()
  toks = f.readline().split()
  for j in xrange(5):
    d[0, j] = float(toks[j])

  return K, d

class DroneAI():

  def __init__(self):
    print "starting drone..."
    self.drone = libardrone.ARDrone()
    self.K, self.d = parse_calib('calibration.txt')
    self.marker_frame = np.array([[0, 0, 0],
                            [MARKER_SIZE, 0, 0],
                            [MARKER_SIZE, MARKER_SIZE, 0],
                            [0, MARKER_SIZE, 0]], dtype=np.float32)
    self.ALLOW_FLIGHT = False
    self.record = Queue(1)
    self.frame_queue = Queue(2)
    self.frame_worker = Process(target=self.query_frames)
    self.frame_worker.start()

  def fly_test(self):
    print "taking off..."
    self.drone.takeoff()
    time.sleep(5)
    print "landing..."
    self.drone.land()
    time.sleep(5)

  def stop(self):
    print "Shutting down..."
    self.record.put(True)
    self.drone.land()
    time.sleep(2)
    self.drone.halt()
    print "Done"

  def query_frames(self):
    print "Starting frame worker"
    cap = cv2.VideoCapture("tcp://192.168.1.1:5555")
    while self.record.empty():
      ret, frame = cap.read()
      if frame == None:
        time.sleep(0.1)
        continue
      if self.frame_queue.full():
        self.frame_queue.get()
      self.frame_queue.put(frame)
    cap.release()
    self.frame_queue.close()
    return

  def get_vid_frame(self):
    return cv2.cvtColor(self.frame_queue.get(), cv2.COLOR_BGR2GRAY)

  def get_target_pose(self, frame):
    markers = detect_markers(frame, BASE_MARKER)
    if not markers:
      return None, None
    marker = markers[0]
    raw_points = [marker.contours[i][0] for i in xrange(4)]
    rot_points = []
    for i in xrange(4):
      j = (i - marker.rotation) % 4
      rot_points.append(raw_points[j])
    rot_points = np.array(rot_points, dtype=np.float32)

    _, rvec, tvec = cv2.solvePnP(np.array([self.marker_frame]), np.array([rot_points]), self.K, self.d, flags=cv2.CV_ITERATIVE)

    return rvec, tvec

  def render_target_pose(self, frame, rvec, tvec):
    marker_pts = np.array([[0, 0, 0],
                           [5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 5]], dtype=np.float32)

    projected_pts, _ = cv2.projectPoints(marker_pts, rvec, tvec, self.K, self.d)

    cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[1][0]), (0, 255, 0), 2)
    cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[2][0]), (0, 0, 255), 2)
    cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[3][0]), (255, 0, 0), 2)

    return frame

  def record_images(self, save_dir):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    buf = []
    pygame.init()
    W, H = 640, 480
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    running = True
    recording = False
    while running:
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            self.stop()
            running = False
            recording = False
          elif event.key == pygame.K_RETURN:
            recording = True

      try:
        frame = self.get_vid_frame()
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        if recording:
          buf.append(frame.copy())

        rvec, tvec = self.get_target_pose(frame)
        if rvec != None and tvec != None:
          frame = self.render_target_pose(frame, rvec, tvec)

        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # battery status
        hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (10, 10, 255)
        bat = drone.navdata.get(0, dict()).get('battery', 0)
        f = pygame.font.Font(None, 20)
        hud = f.render('Battery: %i%%' % bat, True, hud_color)

        screen.blit(hud, (10, 10))
      except:
        pass

      pygame.display.flip()
      clock.tick(50)
      pygame.display.set_caption("FPS: %.2f" % clock.get_fps())

    print "Saving images to {0}".format(save_dir)
    for i in xrange(len(buf)):
      img = buf[i]
      path = os.path.join(save_dir, 'img_{0}.jpg'.format(i))
      cv2.imwrite(path, img)


  def run(self):
    pygame.init()
    W, H = 640, 480
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    running = True
    while running:
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            self.stop()
            running = False
          elif event.key == pygame.K_RETURN:
            self.drone.takeoff()
            self.ALLOW_FLIGHT = True
        elif event.type == pygame.KEYUP:
          print "Taking off..."
          self.drone.hover()

      try:
        frame = self.get_vid_frame()
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

        rvec, tvec = self.get_target_pose(frame)
        #print rvec, tvec
        #rvec = None
        #tvec = None
        if rvec != None and tvec != None:
          print self.ALLOW_FLIGHT
          if self.ALLOW_FLIGHT:
            theta = np.arctan2(tvec[0], tvec[2])
            x, y, z, r = 0, 0, 0, 0
            if tvec[0] > np.pi/6:
              r = 0.2
            elif tvec[0] < -np.pi/6:
              r = -0.2
            #if tvec[2] > 50:
            #  z = -0.1
            #elif tvec[0] < 45:
            #  z = 0.1

          print 'sending move command {0} {1} {2} {3}'.format(x, y, z, r)
          self.drone.move_x(x, y, z, r)
          print 'rendering pose'
          frame = self.render_target_pose(frame, rvec, tvec)
          print 'sleeping'
          time.sleep(0.5)
        else:
          self.drone.hover()

        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # battery status
        hud_color = (255, 0, 0) if self.drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (10, 10, 255)
        bat = self.drone.navdata.get(0, dict()).get('battery', 0)
        f = pygame.font.Font(None, 20)
        hud = f.render('Battery: %i%%' % bat, True, hud_color)

        screen.blit(hud, (10, 10))
      except Exception as e:
        print e
        pass

      pygame.display.flip()
      clock.tick(50)
      pygame.display.set_caption("FPS: %.2f" % clock.get_fps())

    pygame.quit()



pilot = DroneAI()
pilot.run()


#save_dir = '/Users/jsha/tmp/drone_imgs'
#pilot.record_images(save_dir)


#test_frame = cv2.imread('/Users/jsha/tmp/test_marker.jpg', cv2.IMREAD_GRAYSCALE)
#print test_frame.shape
#pilot.get_target_pose(test_frame)
#print test_frame.shape
#pilot.fly_test()
