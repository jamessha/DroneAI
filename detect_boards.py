import glob
import os
import sys
from multiprocessing import Pool

from scipy.misc import imread

import pycb

base_dir = 'calib_data/'

#board_preference = [(9,8), (5,4)]
board_preference = [(9,8)]

def write_board(filename, corners, chessboards):
    if len(chessboards) == 0:
        print "WARNING: couldn't find any board in file %s" % (filename)
        return
    board = None
    output_filename = os.path.splitext(filename)[0] + "_corners.txt"
    for board_size in board_preference:
        boards = [x for x in chessboards if x.shape == board_size]
        if len(boards) != 0:
            board = boards[0]
            break
    if board is None:
        print "WARNING: couldn't find board with right size in file %s. Had " \
                "boards with sizes: " % (filename)
        for board in chessboards:
            print "(%d, %d)" % (board.shape[0], board.shape[1])
        return
    else:
        print "Found board in file %s" % filename
        sys.stdout.flush()
    board_points = corners[board.flatten()]
    file = open(output_filename, 'w')
    for point in board_points:
        file.write("%f, %f\n" % (point[0], point[1]))
    file.close()

def warn(filename):
    print "WARNING: Couldn't load image in file %s" % filename

def detect_board(filename):
    output_filename = os.path.splitext(filename)[0] + "_corners.txt"
    if os.path.exists(output_filename):
        print "Skipping %s, exists." % filename
        return True, filename
    else:
        print "Detecting on %s" % filename
    thresh = True
    # This may no longer be necessary.
    #if 'bmp' in filename:
    #    thresh = False
    try:
        img = imread(filename)
    except:
        warn(filename)
        return False, filename
    if len(img.shape) == 0:
        warn(filename)
        return False, filename
    try:
        corners, chessboards = pycb.extract_chessboards(img, use_corner_thresholding=thresh)
    except Exception as e:
        print "Exception on file", filename
        raise e
    if len(chessboards) > 0:
        write_board(filename, corners, chessboards)
        return True, filename
    else:
        return False, filename

if __name__ == "__main__":
    files = glob.glob("%s/*.jpg" % base_dir)
    pool = Pool(20)
    results = pool.map(detect_board, files)
    for success, filename in results:
        if not success:
            print "Failed on %s" % filename
