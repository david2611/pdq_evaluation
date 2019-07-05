import read_files
import argparse

parser = argparse.ArgumentParser('Convert COCO format detection .json file to an RVC1 format detection .json file')
parser.add_argument('--coco_gt', help='coco format ground-truth .json file')
parser.add_argument('--coco_det', help='coco format detections .json file')
parser.add_argument('--rvc1_det', help='where to save .json file in converted rvc1 format')
args = parser.parse_args()

read_files.convert_coco_det_to_rvc_det(args.coco_det, args.coco_gt, args.rvc1_det)

