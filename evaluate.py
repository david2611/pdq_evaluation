import read_files
import argparse
import glob
import os
import sys
from pdq import PDQ
from coco_mAP import coco_mAP
import json
import numpy as np
import rvc1_gt_loader
import rvc1_class_list
import rvc1_submission_loader

_NUM_VALID = 4

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', default='coco', help='define if we are testing on coco or rvc1 data')
parser.add_argument('--save_folder', help='define where evaluation results will be saved. This includes '
                                          'score.txt with summary and details.json which holds detailed'
                                          'analysis values')
parser.add_argument('--det_folder', help='define where detection .json files are held (one for each sequence if rvc1'
                                         'or one if coco)')
parser.add_argument('--set_cov', type=float, help='set covariance for all gt corners')
parser.add_argument('--mAP_heatmap', action='store_true', help='flag for dictating that mAP should be calculated using'
                                                               'outskirts of heatmap rather than the box corner locations')
args = parser.parse_args()

# Define these before using this code
coco_gt_file = '/mnt/storage_device/Datasets/COCO/annotations/instances_val2014.json'
rvc1_gt_folder = '/mnt/storage_device/Datasets/rvc_challenge1/validation_data/ground_truth'


class ParamSequenceHolder:
    def __init__(self, gt_instances_lists, det_instances_lists, filter_gt):
        """
        Hacky short version of match_sequences from codalab challenge (fewer safety checks)
        :param gt_instances_lists: list of gt_instance_lists (one gt_instance_list per sequence)
        :param det_instances_lists: list of det_instance_lists (one det_instance_list per sequence)
        Note, order of gt_instances_list and det_instances_list must be the same (corresponding sequences)
        :param filter_gt: boolean describing if gt objects should be filtered by size (used for rvc1 only)
        """
        self._gt_instances_lists = gt_instances_lists
        self._det_instances_lists = det_instances_lists
        self._filter_gt = filter_gt

    def __len__(self):
        length = np.sum([len(gt_list) for gt_list in self._gt_instances_lists])
        return length

    def __iter__(self):
        for idx in range(len(self._gt_instances_lists)):
            gt_list = self._gt_instances_lists[idx]
            det_list = self._det_instances_lists[idx]
            if len(gt_list) != len(det_list):
                print("ERROR! gt_list and det_list for sequence {0} not the same length".format(idx))
                print("len GT: {0}\nlen Det: {1}".format(len(gt_list), len(det_list)))
                sys.exit("ERROR! gt_list and det_list for sequence {0} not the same length".format(idx))
            for frame_gt, frame_detections in zip(gt_list, det_list):
                ground_truth = list(frame_gt)
                detections = list(frame_detections)
                yield ground_truth, detections, self._filter_gt


def gen_param_sequence():
    # Load GTs and Detections as appropriate for different data sets (multiple sequences or one folder)
    if args.test_set == 'coco':
        # output is a generator of lists of GTInstance objects and a map of gt_class_ids
        gt_instances, gt_class_ids = read_files.read_COCO_gt(coco_gt_file, ret_classes=True, n_imgs=100)
        det_filenames = glob.glob(os.path.join(args.det_folder, '*.json'))
        if len(det_filenames) != 1:
            sys.exit("ERROR! Invalid number of .json results found in det_folder for coco tests")

        # output is a generator of lists of DetectionInstance objects (BBox or PBox depending)
        det_instances = read_files.read_pbox_json(det_filenames[0], gt_class_ids, n_imgs=100)
        all_gt_instances = [gt_instances]
        all_det_instances = [det_instances]
        filter_gt = False

    elif args.test_set == 'rvc1':
        # output is a list of generator of generators or GTInstance objects
        all_gt_instances = rvc1_gt_loader.read_ground_truth(rvc1_gt_folder)
        all_det_instances = rvc1_submission_loader.read_submission(args.det_folder,
                                                                   ["{0:06d}".format(idx) for idx in range(_NUM_VALID)])
        # all_det_instances = []
        # for det_filename in sorted(glob.glob(os.path.join(args.det_folder, "*.json"))):
        #     # output is generator of lists of DetectionInstance objects (BBox or PBox depending)
        #     all_det_instances.append(read_files.read_pbox_json(det_filename, rvc1_class_list.CLASS_IDS))
        filter_gt = True

    else:
        sys.exit("ERROR! Invalid test_set parameter (must be 'coco' or 'rvc1')")

    param_sequence = ParamSequenceHolder(all_gt_instances, all_det_instances, filter_gt)

    return param_sequence


def main():
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    print("Extracting GT and Detections")
    param_sequence = gen_param_sequence()

    print("Calculating PDQ")
    # Get summary statistics (PDQ, avg_qualities, mAP)
    evaluator = PDQ()
    score = evaluator.score(param_sequence)
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()

    # TODO check mAP is actually working as expected (same results as would get normally)
    print("Calculating mAP")
    if args.mAP_heatmap:
        mAP = coco_mAP(param_sequence, use_heatmap=True)
    else:
        mAP = coco_mAP(param_sequence, use_heatmap=False)

    result = {"score": score*100, "avg_pPDQ": avg_overall_quality, "avg_spatial": avg_spatial_quality,
              "avg_label": avg_label_quality, "TP": TP, "FP": FP, "FN": FN, 'mAP': mAP}
    print("PDQ: {0:4f}\n"
          "mAP: {1:4f}\n"
          "avg_pPDQ:{2:4f}\n"
          "avg_spatial:{3:4f}\n"
          "avg_label:{4:4f}\n"
          "TP:{5}\nFP:{6}\nFN:{7}".format(score, mAP, avg_overall_quality, avg_spatial_quality,
                                          avg_label_quality, TP, FP, FN))

    with open(os.path.join(args.save_folder, 'scores.txt'), 'w') as output_file:
        output_file.write("\n".join("{0}:{1}".format(k, v) for k, v in result.items()))

    # Get the detection-wise and ground-truth-wise qualities and matches for PDQ and save them to file
    all_gt_eval_dicts = evaluator._gt_evals
    all_det_eval_dicts = evaluator._det_evals
    # TODO sort these out to match their individual sequences if possible
    with open(os.path.join(args.save_folder, 'gt_eval_stats.json'), 'w') as f:
        json.dump(all_gt_eval_dicts, f)
    with open(os.path.join(args.save_folder, 'det_eval_stats.json'), 'w') as f:
        json.dump(all_det_eval_dicts, f)


if __name__ == '__main__':
    main()