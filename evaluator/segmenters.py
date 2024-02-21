from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt



class FlowSegmenter():
    """Interface an optical flow predictor to a ClipEvaluator.

    """

    def __init__(
            self, flow_predictor=None, flow_predictor_prob=None, debug=False):
        """Initialize a FlowSegmenter object.

        Exactly one of flow_predictor and flow_predictor_prob arguments
        has to be set (not None).

        Keyword arguments:
        flow_predictor -- an initialized optical flow predictor object
            which has a flow function
        flow_predictor_prob -- an initialized probabilistic optical flow
            predictor object which has a flow function
        debug -- debug mode, show the predicted motion masks (default False)
        """
        assert(
            bool(flow_predictor is None) ^ bool(flow_predictor_prob is None))
        self.of_prob = flow_predictor_prob
        self.of = flow_predictor
        self.img_ref = None
        self.debug = debug


    def next_image(self,rgb,thr_motion,set_reference,actor_mask=None):
        """Return a motion mask, optical flow and uncertainty.

        Arguments:
        rgb -- an RGB image (np.array)
        thr_motion -- flow magnitude threshold for segmentation (float)
        set_reference -- If True, the given rgb image should be set as the
            reference image for optical flow motion prediction.
        actor_mask -- If not None, use this 2D binary mask (np.array) to
            suppress motion and flow predicted at the pixels where the
            actor_mask is True. (default None)

        Return:
        moving -- a 2D binary mask which is True only at predicted moving
            pixels (np.array)
        flow -- predicted optical flow (np.array)
        flow_var -- predicted optical flow uncertainty (np.array)
            If the chosen optical flow method is not probabilistic,
            flow_var is None.
        """
        MAX_VAR = np.inf  #4 1
        if self.img_ref is None:
            assert(set_reference)
        if set_reference:
            self.img_ref = deepcopy(rgb)
        if self.of_prob is not None:
            res = self.of_prob.flow(rgb,self.img_ref)
            flow, flow_var = res[0:2]
        else:
            res = self.of.flow(rgb,self.img_ref)
            flow = res[0]
            flow_var = None
        mag = np.linalg.norm(flow[...,0:2],axis=2)
        moving = mag > thr_motion
        if self.of_prob is not None:
            var_mag = np.linalg.norm(flow_var, axis=-1)
            uncertain = var_mag > MAX_VAR
            moving[uncertain] = False
        if actor_mask is not None:
            moving[actor_mask] = False
            flow[actor_mask,:] = 0
            if flow_var is not None:
                flow_var[actor_mask,:] = 0
        if self.debug:
            plt.imshow(moving)
            plt.show()
        return moving, flow, flow_var

