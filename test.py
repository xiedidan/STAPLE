from StapleWrapper import Staple
import numpy as np
import cv2

image_file = './sequence/00000001.jpg'

if __name__ == '__main__':
    # read image
    image = cv2.imread(image_file)

    tracker = Staple()
    tracker.init(image, np.array([10., 20., 30., 40.], dtype=float))
    
    location = tracker.update(image)
    print(location)
    