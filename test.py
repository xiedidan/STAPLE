import time

from StapleWrapper import Staple
import numpy as np
import cv2

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = Axes3D(fig)

image_path = './sequence/'
seq_len = 105
init_position = [492.000, 417.000, 539.000, 463.000]
response_position = 50

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
                response = tracker.response
                print(response.shape)
                X = np.arange(0, response.shape[0], 1)
                Y = np.arange(0, response.shape[1], 1)
                X, Y = np.meshgrid(X, Y)
                ax.plot_surface(X, Y, response, rstride=5, cstride=5, cmap='viridis')
                # imshow(response)
                show()

        toc = time.time() - tic
        python_total_time += toc

    total_time = tracker.time
    avg_time = total_time / seq_len
    fps = seq_len / total_time

    pfps = seq_len / python_total_time

    print("Total time: {}, total len: {}, avg time: {}, FPS: {}, PTotal Time: {}, PFPS: {}".format(total_time, seq_len, avg_time, fps, python_total_time, pfps))