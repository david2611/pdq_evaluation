"""
Code requires use of LRP code available at the following link: https://github.com/cancam/LRP.
As currently set-up, after downloading and installing the code, change pycocotools folder name to pycocotools_lrp to avoid
confusion with pycocotools from mAP COCO code.
Must append path to LRP code to system path.
"""
from convert_to_coco_input import generate_coco_ground_truth_and_detections
import sys

# Temp way to get access to COCO code for now
sys.path.append('/mnt/storage_device/postdoc_2018-2020/projects/LRP_evaluation/LRP/cocoLRPapi-master/PythonAPI/')
from pycocotools_lrp.cocoevalLRP import COCOevalLRP
from pycocotools_lrp.coco import COCO


def coco_LRP(param_sequence, use_heatmap=True, full=False):
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
