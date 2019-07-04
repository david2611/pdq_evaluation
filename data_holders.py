from __future__ import absolute_import, division, print_function, unicode_literals
import utils
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import os.path as osp

_HEATMAP_THRESH = 0.0027
_2D_MAH_DIST_THRESH = 3.439
_SMALL_VAL = 1e-14


class GroundTruthInstance(object):
    def __init__(self, segmentation_mask, true_class_label, coco_bounding_box=None, num_pixels=None,
                 coco_ignore=False, coco_iscrowd=False, coco_area=None):
        """
        GroundTruthInstance object initialisation function. GroundTruthInstance objects are designed to hold
        data pertinent for a ground-truth object in the context of evaluation. Namely, the segmentation mask,
        its class label, its bounding box and number of pixels. To ensure accurate conversion from our format
        back to COCO format, we also store (where possible) the bounding box, area, and ignore and iscrowd flags
        provided by COCO formatted ground-truths.
        :param segmentation_mask: Boolean 2D array of pixels defining the pixels which make-up the ground-truth object
        :param true_class_label: integer dictating the class label (value between 0 and 1 - <number_of_classes>)
        :param coco_bounding_box: bounding box provided by COCO annotations (but converted to [x1, y1, x2, y2] format)
        (default None)
        :param num_pixels: number of pixels in detection, calculated from segmentation_mask if not provided (default None)
        :param coco_ignore: Boolean defining if object is to be ignored under COCO annotation
        :param coco_iscrowd: Boolean defining if object is labelled as crowd under COCO annotation
        :param coco_area: Float defining area of ground-truth object as defined under COCO annotation
        """
        self.segmentation_mask = segmentation_mask
        self.class_label = true_class_label
        self.coco_ignore = coco_ignore
        self.coco_iscrowd = coco_iscrowd
        self.coco_area = coco_area
        self.bounding_box = utils.generate_bounding_box_from_mask(segmentation_mask)

        # If coco_bounding_box is not supplied, equate it to the provided bounding_box
        if coco_bounding_box is not None and len(coco_bounding_box) > 0:
            self.coco_bounding_box = coco_bounding_box
        else:
            self.coco_bounding_box = self.bounding_box.copy()

        # Calculate the number of pixels based on segmentation mask if unprovided at initialisation
        if num_pixels is not None and num_pixels > 0:
            self.num_pixels = num_pixels
        else:
            self.num_pixels = np.count_nonzero(segmentation_mask)

        # ensure that a coco area is provided if not explicitly
        if self.coco_area is None:
            self.coco_area = self.num_pixels


class DetectionInstance(object):
    def __init__(self, class_list, heatmap=None):
        """
        DetectionInstance object class for when spatial probability heatmap is readily available and does not need to
        be calculated.
        :param class_list: list of label probabilities for each class, ordered to match the class labelling convention
        of corresponding ground-truth data.
        :param heatmap: 2D float image containing the spatial probability that each pixel is part of the
        detected object.
        """
        self._heatmap = heatmap
        self.class_list = class_list

    def calc_heatmap(self, img_size):
        """
        Function for returning the spatial probability heatmap image of the detection.
        :param img_size: size of the image expected from the heatmap output.
        :return: spatial probability heatmap of the size <img_size>
        """
        return self._heatmap

    def get_max_class(self):
        """
        Function for returning the maximum class id of the detection (index of maximum score in class list)
        :return: maximum class id of the detection.
        """
        return np.argmax(self.class_list)

    def get_max_score(self):
        """
        Function for returning the maximum class score of the detection
        :return: maximum class score of the detection
        """
        return np.amax(self.class_list)


class ProbSegDetInst(DetectionInstance):
    def __init__(self, class_list, detection_file, mask_id, mask_root='', chosen_label=None, box=None):
        """
        Initialisation function for detection instance where full instance heatmap is available on file
        :param class_list: list of label probabilities for each class, ordered to match the class labelling convention
        of corresponding ground-truth data.
        :param detection_file: location of the .npy file containing the heatmap masks for all detections within an image
        :param mask_id: the id of the mask within the detection file that corresponds with this instance
        :param mask_root: root location for all mask files (defaults to current directory)
        :param chosen_label: the chosen label for the detection (optional). If None will be the maximum label in class
        list. Note this is only used for mAP and moLRP, PDQ uses class_list.
        :param box: bounding box to be used for box-style detection analysis like mAP and moLRP (optional).
        If None box will be made based on provided probabilistic segmentation heatmap (probability > _HEATMAP_THRESH).
        """
        super(ProbSegDetInst, self).__init__(class_list)
        self.detection_file = detection_file
        self.mask_id = mask_id
        self.mask_root = mask_root
        if chosen_label is None:
            self.chosen_label = np.argmax(self.class_list)
        else:
            self.chosen_label = chosen_label

        if box is None:
            self.box = utils.generate_bounding_box_from_mask(self.calc_heatmap() > _HEATMAP_THRESH)
        self.box = box

    def calc_heatmap(self, img_size=None):
        """
        Create heatmap from stored file. Assumes .npy file stores all heatmaps in numpy array m x h x w.
        Where m is number of heatmap masks, h is image height, and w is image width.
        Numpy array is assumed to be float32 format with values between 0 and 1 or uint8 format with values between
        0 and 255.
        :param img_size: size of the image expected from the heatmap output. Default here is None as image size should
        come from the loaded file. If img_size is provided, it is used to check heatmap output matches what is expected.
        :return: spatial probability heatmap of the size <img_size>
        """
        all_heatmaps = np.load(osp.join(self.mask_root, self.detection_file))

        if len(all_heatmaps.shape) > 2:
            heatmap = all_heatmaps[self.mask_id]
        elif self.mask_id == 0:
            heatmap = all_heatmaps
        else:
            print('ERROR! Heatmap image has only 2 dimensions and mask_id > 0')
            return None

        # Convert to float dtype if not already
        if heatmap.dtype == np.uint8:
            heatmap = heatmap.astype(np.float32)
            heatmap /= 255

        if img_size is not None:
            # Check heatmap sizes are matching
            if heatmap.shape != img_size:
                print("ERROR! Image size does not match heatmap on file")
                return None

        return heatmap


class BBoxDetInst(DetectionInstance):
    def __init__(self, class_list, box, pos_prob=1.0):
        """
        Initialisation function for a bounding box (BBox) detection instance
        :param class_list: list of label probabilities for each class, ordered to match the class labelling convention
        of corresponding ground-truth data.
        :param box: list of box corners that contain the object detected (inclusively).
        Formatted [x1, y1, x2, y2]
        :param pos_prob: float depicting the positional confidence
        """
        super(BBoxDetInst, self).__init__(class_list)
        self.box = box
        self.pos_prob = pos_prob

    def calc_heatmap(self, img_size):
        """
        Function for returning the spatial probability heatmap of the detection
        :param img_size: size of the image expected from the heatmap output
        :return: spatial probability heatmap of the size <img_size>
        """

        heatmap = np.zeros(img_size, dtype=np.float32)
        x1, y1, x2, y2 = self.box
        x1_c, y1_c = np.ceil(self.box[0:2]).astype(np.int)
        x2_f, y2_f = np.floor(self.box[2:]).astype(np.int)
        # Even if the cordinates are integers, there should always be a range
        x1_f = x1_c - 1
        y1_f = y1_c - 1
        x2_c = x2_f + 1
        y2_c = y2_f + 1

        heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]),
                max(x1_f, 0):min(x2_c + 1, img_size[1])] = self.pos_prob
        if y1_f >= 0:
            heatmap[y1_f, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y1_c - y1
        if y2_c < img_size[0]:
            heatmap[y2_c, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y2 - y2_f
        if x1_f >= 0:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x1_f] *= x1_c - x1
        if x2_c < img_size[1]:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x2_c] *= x2 - x2_f
        return heatmap


class PBoxDetInst(DetectionInstance):
    def __init__(self, class_list, box, covs):
        """
        Initialisation function for a probabilistic bounding box (PBox) detection instance
        :param class_list: list of label probabilities for each class, ordered to match the class labelling convention
        of corresponding ground-truth data.
        :param box: list of Gaussian corner mean locations, formatted [x1, y1, x2, y2].
        :param covs: list of two 2D covariance matrices used to define the covariances of the Gaussian corners.
        Formatted [cov1, cov2] where cov1 and cov2 are formatted [[variance_x, corr], [corr, variance_y]]
        """
        super(PBoxDetInst, self).__init__(class_list)
        self.box = box
        self.covs = covs

    def calc_heatmap(self, img_size):
        """
        Function for returning the spatial probability heatmap of the detection
        :param img_size: size of the image expected from the heatmap output
        :return: spatial probability heatmap of the size <img_size>
        """
        # get all covs in format (y,x) to match matrix ordering
        covs2 = [np.flipud(np.fliplr(cov)) for cov in self.covs]

        prob1 = gen_single_heatmap(img_size, [self.box[1], self.box[0]], covs2[0])
        prob2 = gen_single_heatmap(img_size, [img_size[0] - (self.box[3] + 1), img_size[1] - (self.box[2] + 1)],
                                   np.array(covs2[1]).T)
        # flip left-right and up-down to provide probability in from bottom-right corner
        prob2 = np.fliplr(np.flipud(prob2))

        # generate final heatmap
        heatmap = prob1 * prob2

        # Hack to enforce that there are no pixels with probs greater than 1 due to floating point errors
        heatmap[heatmap > 1] = 1

        heatmap[heatmap < _HEATMAP_THRESH] = 0

        return heatmap


def find_roi(img_size, mean, cov):
    """
    Function for finding the region of interest for a probability heatmap generated by a Gaussian corner.
    This region of interest is the area with most change therein, with probabilities above 0.0027 and below 0.9973
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[variance_y, corr], [corr, variance_x]] describes the covariance of the
    Gaussian corner.
    :return: roi_box formatted [x1, y1, x2, y2] depicting the corners of the region of interest (inclusive)
    """

    # Calculate approximate ROI
    stdy = cov[0, 0] ** 0.5
    stdx = cov[1, 1] ** 0.5

    minx = int(max(mean[1] - stdx * 5, 0))
    miny = int(max(mean[0] - stdy * 5, 0))
    maxx = int(min(mean[1] + stdx * 5, img_size[1] - 1))
    maxy = int(min(mean[0] + stdy * 5, img_size[0] - 1))

    # If the covariance is singular, we can't do any better in our estimate.
    if np.abs(np.linalg.det(cov)) < 1e-8:
        return minx, miny, maxx, maxy

    # produce list of positions [y,x] to compare to the given mean location
    approx_roi_shape = (maxy + 1 - miny, maxx + 1 - minx)
    positions = np.indices(approx_roi_shape).T.reshape(-1, 2)
    positions[:, 0] += miny
    positions[:, 1] += minx
    # Calculate the mahalanobis distances to those locations (number of standard deviations)
    # Can only do this for non-singular matrices
    mdists = cdist(positions, np.array([mean]), metric='mahalanobis', VI=np.linalg.inv(cov))
    mdists = mdists.reshape(approx_roi_shape[1], approx_roi_shape[0]).T

    # Shift around the mean to change which corner of the pixel we're using for the mahalanobis distance
    dist_meany = max(min(int(mean[0] - miny), img_size[0] - 1), 0)
    dist_meanx = max(min(int(mean[1] - minx), img_size[1] - 1), 0)
    if 0 < dist_meany < img_size[0] - 1:
        mdists[:dist_meany, :] = mdists[1:dist_meany + 1, :]
    if 0 < dist_meanx < img_size[1] - 1:
        mdists[:, :dist_meanx] = mdists[:, 1:dist_meanx + 1]

    # Mask out samples that are outside the desired distance (extremely low probability points)
    mask = mdists <= _2D_MAH_DIST_THRESH
    mask[dist_meany, dist_meanx] = True  # Force the pixel containing the mean to be true, we always care about that
    roi_box = utils.generate_bounding_box_from_mask(mask)

    return roi_box[0] + minx, roi_box[1] + miny, roi_box[2] + minx, roi_box[3] + miny


def gen_single_heatmap(img_size, mean, cov):
    """
    Function for generating the heatmap for a given Gaussian corner.
    Note that this is a fast approximation and not 100% accurate
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[var_y, corr], [corr, var_x]] describes the covariance of the Gaussian corner.
    :return: heatmap image of size <img_size> with spatial probabilities between 0 and 1.
    """
    heatmap = np.zeros(img_size, dtype=np.float32)

    # Create the gaussian for the corner described
    g = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    # Identify the region of interest (ROI) within the image where values change most
    roi_box = find_roi(img_size, mean, cov)

    # Calculate the cdf probability within the region of interest
    # Note that we subtract small value on the ROI to avoid fencepost issues with extremely low covariances.
    positions = np.dstack(np.mgrid[roi_box[1] + 1:roi_box[3] + 2, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL
    prob = g.cdf(positions)

    if len(prob.shape) == 1:
        prob.shape = (roi_box[3] + 1 - roi_box[1], roi_box[2] + 1 - roi_box[0])

    # Fill the the heatmap image probabilities as appropriate for approximating.
    # Probabilities within the ROI equal the calculated probabilities
    heatmap[roi_box[1]:roi_box[3]+1, roi_box[0]:roi_box[2]+1] = prob
    # Probabilities to the right of the ROI equal the column values of the extreme right edge of the ROI
    heatmap[roi_box[3]:, roi_box[0]:roi_box[2]+1] = np.array(heatmap[roi_box[3], roi_box[0]:roi_box[2]+1], ndmin=2)
    # Probabilities below the ROI equal the row values of the extreme bottom edge of the ROI
    heatmap[roi_box[1]:roi_box[3]+1, roi_box[2]:] = np.array(heatmap[roi_box[1]:roi_box[3]+1, roi_box[2]], ndmin=2).T
    # Probabilities to the bottom-right of the ROI equals 1
    heatmap[roi_box[3]+1:, roi_box[2]+1:] = 1.0

    # If your region of interest includes outside the main image, remove probability of existing outside the image
    # Remove probability of being outside in the x direction
    if roi_box[0] == 0:
        pos_outside_x = np.dstack(np.mgrid[roi_box[1] + 1:roi_box[3] + 2, 0:1]) - _SMALL_VAL  # points left of the image
        prob_outside_x = np.zeros((img_size[0], 1), dtype=np.float32)
        prob_outside_x[roi_box[1]:roi_box[3] + 1, 0] = g.cdf(pos_outside_x)
        prob_outside_x[roi_box[3] + 1:, 0] = prob_outside_x[roi_box[3], 0]
        # Final probability is your overall cdf minus the probability in-line with that point along
        # the border for both dimensions plus the cdf at (-1, -1) which has points counted twice otherwise
        heatmap -= prob_outside_x

    # Remove probability of being outside in the x direction
    if roi_box[1] == 0:
        pos_outside_y = np.dstack(np.mgrid[0:1, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL  # points above the image
        prob_outside_y = np.zeros((1, img_size[1]), dtype=np.float32)
        prob_outside_y[0, roi_box[0]:roi_box[2] + 1] = g.cdf(pos_outside_y)
        prob_outside_y[0, roi_box[2] + 1:] = prob_outside_y[0, roi_box[2]]
        heatmap -= prob_outside_y

    # If we've subtracted twice, we need to re-add the probability of the far top-left corner
    if roi_box[0] == 0 and roi_box[1] == 0:
        heatmap += g.cdf([[[0 - _SMALL_VAL, 0 - _SMALL_VAL]]])

    heatmap[heatmap < _HEATMAP_THRESH] = 0

    return heatmap
