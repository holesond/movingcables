import time

import numpy as np
import cv2



class OnlineFlow():
    """Online optical flow computation interface to the Farneback's method."""
    def __init__(self,gpu=False):
        if True:
            self.pyr_scale = 0.5
            self.levels = 3
            self.winsize = 15
            self.iterations = 3
            self.poly_n = 5
            self.poly_sigma = 1.2
            self.flags = 0
        if False:
            self.pyr_scale = 0.5
            self.levels = 1
            self.winsize = 3
            self.iterations = 15
            self.poly_n = 3
            self.poly_sigma = 5
            self.flags = 1
        self.poly_n = 7
        self.poly_sigma = 1.5
    
    def flow(self, img1, img2):
        """Return optical flow given a pair of BGR/RGB images."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            self.pyr_scale, self.levels, self.winsize, self.iterations,
            self.poly_n, self.poly_sigma, self.flags)
        return flow, None, None

