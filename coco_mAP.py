"""
Code requires official mAP COCO code available for download and install at the following link:
https://github.com/cocodataset/cocoapi
After code is installed, user must append location of PythonAPI to their system path.
"""
import numpy as np
import sys
from convert_to_coco_input import generate_coco_ground_truth_and_detections

# Temp way to get access to COCO code for now
sys.path.append('./cocoapi/PythonAPI/')
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def coco_mAP(param_sequence, use_heatmap=True):
    """
    Calculate COCO mean average precision (mAP) scores using official COCO evaluation code
    :param param_sequence: A list of tuples where each tuple holds a list of GroundTruthInstances and a list of
    DetectionInstances to use for evaluation. Each image observed is an entry in the main list.
    :param use_heatmap: Boolean flag describing if BBox used for evaluation should be based upon heatmap of detection
    (i.e. fit a bounding box around heatmap segmentation mask). (Default True)
    :param full: Boolean describing if full moLRP outputs are returned (moLRP, moLRPLoc, moLRPFP, moLRPFN).
    If true these are returned in a dictionary, if not only moLRP is returned as a float. (Default False)
    :return: mean average precision (mAP) score for the sequence.
    """
    # Create ground truth COCO object
    coco_gt = COCO()

    # Convert GroundTruthInstance and DetectionInstance objects to coco format
    coco_gt_dict, coco_det_list = generate_coco_ground_truth_and_detections(param_sequence, use_heatmap)

    # Finish creating the coco ground truth object
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # Create detection COCO object
    coco_det = coco_gt.loadRes(coco_det_list)

    # Create COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_det, 'bbox')

    # Run evaluation procedure
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Calculate the mean AP for this experiment (based on code from coco_eval.summarize())
    # Note that I assume the idx for max_dets=100 is 2
    # Note that I assume the idx for area_rng='all' is 0
    precisions = coco_eval.eval['precision'][:, :, :, 0, 2]

    # Print COCO evaluation statistics
    coco_eval.summarize()

    # Check that there are precisions here, if not return zero
    if len(precisions[precisions > -1]) == 0:
        return 0.0
    else:
        return np.mean(precisions[precisions > -1])
