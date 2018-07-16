import time

from StapleWrapper import Staple
import numpy as np
import cv2

from pylab import *

image_path = './sequence/'
seq_len = 105
init_position = [492.000, 417.000, 539.000, 463.000]
response_position = 20

debug = True

if __name__ == '__main__':
    tracker = Staple()
    python_total_time = 0.

    for i in range(seq_len):
        # read image
        image_file = '{}{:0>8d}.jpg'.format(image_path, i + 1)
        image = cv2.imread(image_file)

        tic = time.time()

        if i == 0:
            tracker.init(image, np.array(init_position, dtype=float)) 
        else:
            location = tracker.update(image)

            # draw response
            if (response_position - 1) == i:
                response = tracker.response.reshape((75, 75))
                imshow(response)
                show()

        toc = time.time() - tic
        python_total_time += toc

    total_time = tracker.time
    avg_time = total_time / seq_len
    fps = seq_len / total_time

    pfps = seq_len / python_total_time

    print("Total time: {}, total len: {}, avg time: {}, FPS: {}, PTotal Time: {}, PFPS: {}".format(total_time, seq_len, avg_time, fps, python_total_time, pfps))