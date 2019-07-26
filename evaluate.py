import read_files
import argparse
import os
import sys
from pdq import PDQ
from coco_mAP import coco_mAP
import json
import numpy as np
import rvc1_gt_loader
import rvc1_submission_loader
from coco_LRP import coco_LRP

# Input parameters
parser = argparse.ArgumentParser(description='Perform PDQ, mAP, and moLRP evaluation on either coco or rvc1 data.')
parser.add_argument('--test_set', default='coco', choices=['coco', 'rvc1'],
                    help='define if we are testing on coco or rvc1 data')
parser.add_argument('--gt_loc', help='define where ground truth data (as folder of folders or as single file) is.'
                                     'This includes filename if ground truth is given as a file.'
                                     'Will only be treated as a file if test_set is coco')
parser.add_argument('--det_loc', help='define where detection (as folder of .json files or single file) is.'
                                      'This includes filename if ground truth is given as a file.'
                                      'Will only be treated as a file if test_set is coco')
parser.add_argument('--save_folder', help='define where evaluation results will be saved. This includes '
                                          'score.txt with summary and details.json files which hold detailed analysis '
                                          'values. Sub-folders will be used if multiple sequences tested (rvc1)')
parser.add_argument('--set_cov', type=float, help='set covariance for all gt corners')
parser.add_argument('--mAP_heatmap', action='store_true', help='flag for dictating that mAP should be calculated using'
                                                               'outskirts of heatmap rather than the box corner '
                                                               'locations (not used in papers and not recommended)')
parser.add_argument('--bbox_gt', action='store_true', help='Flag determines if you want to treat GT as bounding boxes'
                                                           'rather than segmentation masks.')
parser.add_argument('--segment_mode', action='store_true', help='This flag indicates that the PDQ should be evaluated'
                                                                'in segment_mode meaning the background is any pixel'
                                                                'outside the GT mask not the GT bounding box.'
                                                                'Note, should only be used for prob_seg at present')
parser.add_argument('--greedy_mode', action='store_true', help='This flag indicates if detection-GT assignment is done '
                                                               'in a greedy fashion(assigned in order of highest pPDQ)')
parser.add_argument('--prob_seg', action='store_true', help='this flag indicates that the detections are probabilistic'
                                                            'segmentations and are formatted as such')
parser.add_argument('--mce_mode', action='store_true', help='Define if label scaling factor is based on maximum '
                                                            'calibration error rather than expected calibration error')
parser.add_argument('--ce_max_class', action='store_true', help='Flag determines if ECE is calculated using only the'
                                                                'max class rather than full prob distribution')
parser.add_argument('--ce_no_fn', action='store_true', help='flag dictating if false negatives should be ignored when'
                                                            'calculating calibration errors.')
args = parser.parse_args()

# Define these before using this code
if args.test_set == 'coco':
    coco_gt_file = args.gt_loc
elif args.test_set == 'rvc1':
    rvc1_gt_folder = args.gt_loc


class ParamSequenceHolder:
    def __init__(self, gt_instances_lists, det_instances_lists):
        """
        Class for holding parameters (GroundTruthInstances etc.) for multiple sequences.
        Based upon match_sequences function from codalab challenge but with fewer checks.
        Link to codalab challenge version: https://github.com/jskinn/rvchallenge-evaluation/blob/master/gt_loader.py
        :param gt_instances_lists: list of gt_instance_lists (one gt_instance_list per sequence)
        :param det_instances_lists: list of det_instance_lists (one det_instance_list per sequence)
        Note, order of gt_instances_list and det_instances_list must be the same (corresponding sequences)

        """
        self._gt_instances_lists = gt_instances_lists
        self._det_instances_lists = det_instances_lists

    def __len__(self):
        length = np.sum([len(gt_list) for gt_list in self._gt_instances_lists])
        return length

    def __iter__(self):
        for idx in range(len(self._gt_instances_lists)):
            gt_list = self._gt_instances_lists[idx]
            det_list = self._det_instances_lists[idx]

            # Check the lists are the same length
            if len(gt_list) != len(det_list):
                raise ValueError('gt_list and det_list for sequence {0} not the same length\n'
                                 'length GT: {1}\n'
                                 'length Det {2}'.format(idx, len(gt_list), len(det_list)))

            for frame_gt, frame_detections in zip(gt_list, det_list):
                ground_truth = list(frame_gt)
                detections = list(frame_detections)
                yield ground_truth, detections


def gen_param_sequence():
    """
    Function for generating the parameter sequence to be used in evaluation procedure.
    Parameter sequence holds all GroundTruthInstances, DetectionInstances, and ground-truth filter flags
    across all sequences.
    :return: param_sequences: ParamSequenceHolder containing all GroundTruthInstances, DetectionInstances,
    and ground-truth filter flags across all sequences being evaluated.
    len_sequences: list of sequence lengths for all sequences being evaluated.
    """

    # Load GTs and Detections as appropriate for different data sets (multiple sequences or one folder)
    if args.test_set == 'coco':
        # output is a generator of lists of GTInstance objects and a map of gt_class_ids
        gt_instances, gt_class_ids = read_files.read_COCO_gt(coco_gt_file, ret_classes=True, bbox_gt=args.bbox_gt)
        det_filename = args.det_loc

        # output is a generator of lists of DetectionInstance objects (BBox or PBox depending on detection)
        det_instances = read_files.read_pbox_json(det_filename, gt_class_ids, override_cov=args.set_cov,
                                                  prob_seg=args.prob_seg)
        all_gt_instances = [gt_instances]
        all_det_instances = [det_instances]

    elif args.test_set == 'rvc1':
        # output is a list of generator of generators of GTInstance objects
        all_gt_instances = rvc1_gt_loader.read_ground_truth(rvc1_gt_folder, bbox_gt=args.bbox_gt)
        all_det_instances = rvc1_submission_loader.read_submission(args.det_loc,
                                                                   ["{0:06d}".format(idx) for idx in
                                                                    range(len(all_gt_instances))],
                                                                   override_cov=args.set_cov)

    else:
        sys.exit("ERROR! Invalid test_set parameter (must be 'coco' or 'rvc1')")

    param_sequence = ParamSequenceHolder(all_gt_instances, all_det_instances)
    len_sequences = [len(all_gt_instances[idx]) for idx in range(len(all_gt_instances))]

    return param_sequence, len_sequences


def main():
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    print("Extracting GT and Detections")
    param_sequence, len_sequences = gen_param_sequence()

    print("Calculating PDQ")

    # Get summary statistics (PDQ, avg_qualities)
    evaluator = PDQ(filter_gts=(args.test_set == 'rvc1'), segment_mode=args.segment_mode, greedy_mode=args.greedy_mode,
                    mce_mode=args.mce_mode, ce_max_class=args.ce_max_class, ce_no_fn=args.ce_no_fn)
    pdq = evaluator.score(param_sequence)
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()
    avg_fg_quality = evaluator.get_avg_fg_quality_score()
    avg_bg_quality = evaluator.get_avg_bg_quality_score()
    label_scaling_factor = evaluator.get_label_scaling_factor()

    # Get the detection-wise and ground-truth-wise qualities and matches for PDQ and save them to file
    all_gt_eval_dicts = evaluator._gt_evals
    all_det_eval_dicts = evaluator._det_evals

    # Calculate mAP
    print("Calculating mAP")
    # generate the parameter sequence again for new tests (generator does not hold onto data once used)
    print("Extracting GT and Detections")
    param_sequence, len_sequences = gen_param_sequence()
    if args.mAP_heatmap:
        mAP = coco_mAP(param_sequence, use_heatmap=True)
        print('mAP: {0}'.format(mAP))
    else:
        mAP = coco_mAP(param_sequence, use_heatmap=False)
        print('mAP: {0}'.format(mAP))

    # Calculate LRP
    print("Calculating LRP")
    # generate the parameter sequence again for new tests (generator does not hold onto data once used)
    print("Extracting GT and Detections")
    param_sequence, len_sequences = gen_param_sequence()
    # Use same BBox definition as would be used for mAP
    # Extract all moLRP statistics
    if args.mAP_heatmap:
        LRP_dict = coco_LRP(param_sequence, use_heatmap=True, full=True)
    else:
        LRP_dict = coco_LRP(param_sequence, use_heatmap=False, full=True)

    # Compile evaluation statistics into a single dictionary
    result = {"PDQ": pdq, "avg_pPDQ": avg_overall_quality, "avg_spatial": avg_spatial_quality,
              'avg_fg': avg_fg_quality, 'avg_bg': avg_bg_quality,
              "avg_label": avg_label_quality, "label_scaling_factor": label_scaling_factor, "TP": TP, "FP": FP,
              "FN": FN, 'mAP': mAP, 'moLRP': LRP_dict['moLRP'], 'moLRPLoc': LRP_dict['moLRPLoc'],
              'moLRPFP': LRP_dict['moLRPFP'], 'moLRPFN': LRP_dict['moLRPFN']}
    print("PDQ: {0:4f}\n"
          "mAP: {1:4f}\n"
          "avg_pPDQ:{2:4f}\n"
          "avg_spatial:{3:4f}\n"
          "avg_label:{4:4f}\n"
          "label_scaling_factor:{5:.4f}\n"
          "avg_foreground:{6:4f}\n"
          "avg_background:{7:4f}\n"
          "TP:{8}\nFP:{9}\nFN:{10}\n"
          "moLRP:{11:4f}\n"
          "moLRPLoc:{12:4f}\n"
          "moLRPFP:{13:4f}\n"
          "moLRPFN:{14:4f}\n".format(pdq, mAP, avg_overall_quality, avg_spatial_quality, avg_label_quality,
                                     label_scaling_factor, avg_fg_quality, avg_bg_quality, TP, FP, FN,
                                     LRP_dict['moLRP'], LRP_dict['moLRPLoc'], LRP_dict['moLRPFP'], LRP_dict['moLRPFN']))

    # Save evaluation statistics to file
    with open(os.path.join(args.save_folder, 'scores.txt'), 'w') as output_file:
        output_file.write("\n".join("{0}:{1}".format(k, v) for k, v in sorted(result.items())))

    # Save pairwise PDQ statistics to file for use in visualisation code (separate file for each sequence)
    prev_idx = 0
    for idx, len_sequence in enumerate(len_sequences):
        seq_gt_eval_dicts = all_gt_eval_dicts[prev_idx:prev_idx+len_sequence]
        seq_det_eval_dicts = all_det_eval_dicts[prev_idx:prev_idx + len_sequence]
        prev_idx += len_sequence

        with open(os.path.join(args.save_folder, 'gt_eval_stats_{:02d}.json'.format(idx)), 'w') as f:
            json.dump(seq_gt_eval_dicts, f)
        with open(os.path.join(args.save_folder, 'det_eval_stats_{:02d}.json').format(idx), 'w') as f:
            json.dump(seq_det_eval_dicts, f)


if __name__ == '__main__':
    main()
