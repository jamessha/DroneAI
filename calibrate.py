import glob
import cv2
import numpy as np

SQUARE_SIZE = 1.7 # in cm
BOARD_SIZE = [8, 9] # x, y

base_dir = 'calib_data'

def read_points(filepath):
  points = []
  f = open(filepath)
  for line in f.readlines():
    toks = line.split(',')
    points.append([float(toks[0]), float(toks[1])])
  return np.array(points, dtype=np.float32)


board_points_3d = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), dtype=np.float32)
for j in xrange(BOARD_SIZE[1]):
  for i in xrange(BOARD_SIZE[0]):
    board_points_3d[j * BOARD_SIZE[0] + i, 0] = i * SQUARE_SIZE
    board_points_3d[j * BOARD_SIZE[0] + i, 1] = j * SQUARE_SIZE
    board_points_3d[j * BOARD_SIZE[0] + i, 2] = 0


training_boards = glob.glob(base_dir + '/*.txt')
board_points_2d = []
for board_file in training_boards:
  board_points_2d.append(read_points(board_file))
board_points_3d = [board_points_3d for x in training_boards]

test_img = cv2.imread(base_dir + '/img_0.jpg')
img_size = (test_img.shape[0], test_img.shape[1])
err, K, d, rvecs, tvecs = cv2.calibrateCamera(board_points_3d, board_points_2d, img_size)

outpath = 'calibration.txt'
f = open(outpath, 'w+')
f.write('{0} {1} {2}\n'.format(K[0, 0], K[0, 1], K[0, 2]))
f.write('{0} {1} {2}\n'.format(K[1, 0], K[1, 1], K[1, 2]))
f.write('{0} {1} {2}\n'.format(K[2, 0], K[2, 1], K[2, 2]))
f.write('\n')
f.write('{0} {1} {2} {3} {4}\n'.format(d[0, 0], d[0, 1], d[0, 2], d[0, 3], d[0, 4]))
f.close()
