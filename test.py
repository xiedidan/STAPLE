import time

from StapleWrapper import Staple
import numpy as np
import cv2

image_path = './sequence/'
seq_len = 105
init_position = [492.000, 417.000, 539.000, 463.000]

debug = True

if __name__ == '__main__':
    tracker = Staple()

    for i in range(seq_len):
        # read image
        image_file = '{}{:0>8d}.jpg'.format(image_path, i + 1)
        image = cv2.imread(image_file)

        if i == 0:
            tracker.init(image, np.array(init_position, dtype=float)) 
        else:
            location = tracker.update(image)

    total_time = tracker.time
    avg_time = total_time / seq_len
    fps = seq_len / total_time

    print("Total time: {}, total len: {}, avg time: {}, FPS: {}".format(total_time, seq_len, avg_time, fps))