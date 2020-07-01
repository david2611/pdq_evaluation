"""
Code requires use of LRP code available at the following link: https://github.com/cancam/LRP.
As currently set up, should copy cocoEvalLRP to joint pycocotools folder with mAP code.
Should append path to pycocotools code to system path.
"""
from convert_to_coco_input import generate_coco_ground_truth_and_detections
import sys

# Temp way to get access to COCO code for now
sys.path.append('./cocoapi/PythonAPI/')
from pycocotools.cocoevalLRP import COCOevalLRP
from pycocotools.coco import COCO


def coco_LRP(param_sequence, use_heatmap=True, full=False):
    """
    Calculate LRP scores using third party LRP code designed to work on COCO data
    :param param_sequence: A list of tuples where each tuple holds a list of GroundTruthInstances and a list of
    DetectionInstances to use for evaluation. Each image observed is an entry in the main list.
    :param use_heatmap: Boolean flag describing if BBox used for evaluation should be based upon heatmap of detection
    (i.e. fit a bounding box around heatmap segmentation mask). (Default True)
    :param full: Boolean describing if full moLRP outputs are returned (moLRP, moLRPLoc, moLRPFP, moLRPFN).
    If true these are returned in a dictionary, if not only moLRP is returned as a float. (Default False)
    :return: moLRP if full is False or dictionary containing moLRP, moLRPLoc, moLRPFP, and moLRPFN with metric names
    as keys of the dictionary.
    """
    # Create ground truth COCO object
    coco_gt = COCO()

    # Convert GroundTruthInstance and DetectionInstance objects to coco format
    coco_gt_dict, coco_det_list = generate_coco_ground_truth_and_detections(param_sequence, use_heatmap)

    # Finish creating the coco ground-truth object
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # Create detection COCO object
    coco_det = coco_gt.loadRes(coco_det_list)

    # Create COCO evaluator
    coco_eval_lrp = COCOevalLRP(coco_gt, coco_det)

    # Run evaluation procedure
    coco_eval_lrp.evaluate()
    coco_eval_lrp.accumulate()

    # coco_eval_lrp.summarize(detailed=0)

    # Return either just the moLRP score or full set of moLRP scores including moLRPLoc, moLRPFP, and moLRPFN
    if not full:
        return coco_eval_lrp.eval['moLRP']
    else:
        return {'moLRP': coco_eval_lrp.eval['moLRP'],
                'moLRPLoc': coco_eval_lrp.eval['moLRPLoc'], 'moLRPFP': coco_eval_lrp.eval['moLRPFP'],
                'moLRPFN': coco_eval_lrp.eval['moLRPFN']}
