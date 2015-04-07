import glob
import cv2
import numpy as np

from collections import Counter
from ar_markers.hamming.detect import detect_markers

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

def get_target_pose(frame, K, d):
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
  marker_frame = np.array([[0, 0, 0],
                            [MARKER_SIZE, 0, 0],
                            [MARKER_SIZE, MARKER_SIZE, 0],
                            [0, MARKER_SIZE, 0]], dtype=np.float32)

  _, rvec, tvec = cv2.solvePnP(np.array([marker_frame]), np.array([rot_points]), K, d, flags=cv2.CV_ITERATIVE)

  return rvec, tvec


def render_target_pose(frame, K, d, rvec, tvec):
  marker_pts = np.array([[0, 0, 0],
                         [5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5]], dtype=np.float32)

  projected_pts, _ = cv2.projectPoints(marker_pts, rvec, tvec, K, d)

  cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[1][0]), (0, 255, 0), 2)
  cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[2][0]), (0, 0, 255), 2)
  cv2.line(frame, tuple(projected_pts[0][0]), tuple(projected_pts[3][0]), (255, 0, 0), 2)

  return frame


K, d = parse_calib('calibration.txt')

BASE_DIR = "/Users/jsha/tmp/drone_imgs"
imgs = glob.glob(BASE_DIR + '/img_*.jpg')

n = 0
for img in imgs:
  frame = cv2.imread(img)
  rvec, tvec = get_target_pose(frame, K, d)
  if rvec == None or tvec == None:
    print "Missing {0}".format(img)
    continue
  n += 1
  #frame = render_target_pose(frame, K, d, rvec, tvec)
  #print "Showing {0}".format(img)
  #cv2.imshow('test', frame)
  #cv2.waitKey(0)

print "found {0}/{1} boards".format(n, len(imgs))
