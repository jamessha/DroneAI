import glob
import os

import numpy as np
from scipy.misc import imread

import pycb

base_dir = 'calib_data'

board_sizes = [(9,8), (5,4)]

def load_board(corners_filename):
    corners = []
    if not os.path.exists(corners_filename):
        return None, None
    f = open(corners_filename)
    for line in f:
        corners.append([float(x.strip()) for x in line.split(",")])
    f.close()
    corners = np.array(corners)
    for board_size in board_sizes:
        if np.prod(board_size) == len(corners):
            chessboard = np.zeros(board_size, dtype=np.int)
            idx = 0
            for r in range(board_size[0]):
                for c in range(board_size[1]):
                    chessboard[r, c] = idx
                    idx += 1
    return corners, chessboard

def show_board(img_filename):
    corners_filename = os.path.splitext(img_filename)[0] + "_corners.txt"
    corners, chessboard = load_board(corners_filename)
    if corners is None:
        return
    img = imread(img_filename)
    pycb.draw_boards(img, corners, [chessboard])

if __name__ == "__main__":
    files = sorted(glob.glob("%s/*.jpg" % base_dir))
    print files
    #files = [x for x in files if "high" in x]
    for file in files:
        print file
        show_board(file)
