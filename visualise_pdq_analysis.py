import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import read_files
import glob
import sys
import rvc1_gt_loader
import rvc1_submission_loader
import rvc1_class_list
import os
import cv2
from copy import copy
import matplotlib.patches as patches
import utils
from data_holders import PBoxDetInst
from tqdm import tqdm

# Input parameters
parser = argparse.ArgumentParser(description='Visualise probabilistic detections and PDQ analysis for a single '
                                             'sequence of images.')
parser.add_argument('--data_type', choices=['coco', 'rvc1'], help='type of data being evaluated')
parser.add_argument('--ground_truth', help='file or folder location where ground-truth is kept')
parser.add_argument('--gt_img_folder', help='folder with all gt images in order of gt_instances')
parser.add_argument('--det_json', help='filename for detection file to be matched with the ground-truth')
parser.add_argument('--det_analysis', help='filename for json containing detection-wise analysis information')
parser.add_argument('--gt_analysis', help='filename for json containing ground-truth-wise analysis information')
parser.add_argument('--save_folder', help='location where all analysis images will be stored')
parser.add_argument('--set_cov', type=float, help='set covariance for all det corners')
parser.add_argument('--img_type', help='type of image in gt_img_folder (png or jpg)')
parser.add_argument('--full_info', action='store_true', help='flag for stating if all pPDQ information should be'
                                                             'displayed as part of the figure.')
parser.add_argument('--img_set', nargs='+', help='list of img files to create visualisations for. '
                                                 'Note that if none is provided, all images are used')
parser.add_argument('--colour_mode', choices=['gr', 'bo'], default='bo',
                    help='Dictate which colour mode you wish to use. gr = green correct, red incorrect.'
                         'bo = blue correct orange incorrect.')
parser.add_argument('--corner_mode', default='ellipse', choices=['arrow', 'ellipse'],
                    help='what method for drawing corners is to be used')
args = parser.parse_args()

# Create save folder
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)

# Define the colour-scheme to be used in visualisations
if args.colour_mode == 'bo':
    correct_colour = 'blue'
    incorrect_colour = 'C1'     # Orange
else:
    correct_colour = 'green'
    incorrect_colour = 'red'

# Define palette for gt mask images
# incorrect for false negatives (above 1.0)
# correct for true positives (below 1.0)
# transparent for normal pixels
palette = copy(plt.cm.gray)
palette.set_over(incorrect_colour, 1.0)
palette.set_under(correct_colour, 1.0)
palette.set_bad(alpha=0.0)

# Set a font size for text in boxes
_FONTSIZE = 16


def load_gt_and_det_data(gt_loc, det_json, data_type):
    """
    Function for loading ground-truth and detection instances for a single folder/sequence
    :param gt_loc: location of folder/file where ground-truth information can be extracted
    :param det_json: detection json file for the given folder/sequence
    :param data_type: string defining if we are analysing 'coco' or 'rvc1' data as this will effect
    what, and how, data is loaded given the specific filtering process of rvc1.
    :return: gt_instances, det_instances, and class list
    """
    if data_type == 'coco':
        # output is a generator of lists of GroundTruthInstance objects and a map of gt_class_ids
        gt_instances, gt_class_ids_map = read_files.read_COCO_gt(gt_loc, ret_classes=True)

        # output is a generator of lists of DetectionInstance objects (BBox or PBox depending)
        det_instances = read_files.read_pbox_json(det_json, gt_class_ids_map, override_cov=args.set_cov)
        class_idxs = [gt_class_ids_map[key] for key in sorted(gt_class_ids_map.keys())]
        class_names = list(sorted(gt_class_ids_map.keys()))
        class_list = [class_names[idx] for idx in np.argsort(class_idxs)]
    elif data_type == 'rvc1':
        gt_instances = rvc1_gt_loader.SequenceGTLoader(gt_loc)
        det_instances = rvc1_submission_loader.DetSequenceLoader(det_json)
        class_list = rvc1_class_list.CLASSES
    else:
        sys.exit("ERROR! Invalid data type provided")

    # check gt_instances and det_instances are the same size
    if len(gt_instances) != len(det_instances):
        sys.exit("ERROR! gt and det not same length. gt: {0}, det: {1}".format(len(gt_instances), len(det_instances)))

    return gt_instances, det_instances, class_list


def save_analysis_img(img_name, img_gts, img_dets, img_gt_analysis, img_det_analysis, class_list, save_folder,
                      full_info, corner_mode):
    """
    Generate and save an analysis visualisation image.
    In the image, all ground-truth segmentation masks are overlayed on their objects, class name written in the centre
    of the object, and all detection boxes, with visualisation of covariance corners are drawn.
    TPs have segmentation masks and detections are given "correct colour" (green or blue) and
    optionally pairwise quality measures (pPDQ, spatial quality, label quality, max_label).
    FPs and FNs are given "incorrect colour" (red or orange).
    FPs provide the maximum non-background class label and label confidence in top-left of box.
    :param img_name: Full name of the image to have visualisations drawn upon
    :param img_gts: list/generator of GroundTruthInstances for the given image
    :param img_dets: list/generator of DetectionInstances for the given image
    :param img_gt_analysis: list/generator of analysis dictionaries that correspond to the GroundTruthInstances of
    the given image and how they correspond to the DetectionInstances of the image
    :param img_det_analysis: list/generator of analysis dictionaries that correspond to the DetectionInstances of
    the given image and how they correspond to the GroundTruthInstances of the image
    :param class_list: ordered class list matching the ordering of ground-truth and detection labelling conventions
    :param save_folder: folder where image with visualisations will be saved
    :param full_info: Boolean dictating if full pairwise quality information will be shown in visualisation for TPs.
    If false, only detection maximum class + maximum class confidence will be provided alongside each detection
    :param corner_mode: Either 'arrow' or 'ellipse' depending on which format Gaussian corners shall be visualised.
    :return: None
    """
    #  get base image
    img = cv2.imread(img_name)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    ratio = img.shape[0]/float(img.shape[1])
    # Set savefile image to be 12 inches max dimension for clarity
    if ratio <= 1:
        fig_size = (12, 12*ratio)
    else:
        fig_size = (12*(1/ratio), 12)
    fig = plt.figure(figsize=fig_size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(img)

    # Add gt segmentation masks blended onto the image
    for gt_idx, (gt_inst, gt_analysis) in enumerate(zip(img_gts, img_gt_analysis)):
        # skip if gt was ignored at analysis time
        if gt_analysis['ignore']:
            continue

        mask = gt_inst.segmentation_mask.astype(int)
        # set mask to correct value for TP or FN
        if gt_analysis['matched']:
            text_box_colour = correct_colour
            mask[mask > 0] = -1
        else:
            text_box_colour = incorrect_colour
            mask[mask > 0] = 2

        # Draw mask image
        img_mask = np.ma.masked_where(mask == 0, mask)
        ax.imshow(img_mask, cmap=palette, interpolation='none', alpha=0.35, vmax=1.0, vmin=0.0)

        # Add class label at in centre of mask
        label = class_list[gt_inst.class_label]
        gt_box = gt_inst.bounding_box
        textx = ((gt_box[2] - gt_box[0])/2.)+gt_box[0]
        texty = ((gt_box[3] - gt_box[1]) / 2.) + gt_box[1]
        label_string = '({:d}) {:s}'.format(gt_idx, label)
        ax.text(textx, texty, label_string, horizontalalignment='center',
                verticalalignment='center', bbox=dict(facecolor=text_box_colour, alpha=0.3), fontsize=_FONTSIZE)

    # add detection boxes to the image
    for det_idx, (det_inst, det_analysis) in enumerate(zip(img_dets, img_det_analysis)):
        # skip if det was ignored at analysis time
        if det_analysis['ignore']:
            continue

        det_box = det_inst.box

        if det_analysis['matched']:
            colour = correct_colour
        else:
            colour = incorrect_colour
        ax.add_patch(
            patches.Rectangle((det_box[0], det_box[1]), (det_box[2] - det_box[0]) + 1, (det_box[3] - det_box[1]) + 1,
                              edgecolor=colour, facecolor='none', linewidth=3))
        # draw covariances
        if isinstance(det_inst, PBoxDetInst):
            det_covs = det_inst.covs
            utils.draw_cov(det_box, det_covs, ax, colour=colour, mode=corner_mode)

        # Write text
        if det_analysis['matched'] and full_info:
            # Provide pairwise analysis statistics if asked for them and matched
            correct_class = class_list[det_analysis['correct_class']]
            max_class = class_list[np.argmax(det_inst.class_list)]
            det_str = '[{0}]: {1} pPDQ: {2:.3f}' \
                      '\nspatial: {3:.3f}' \
                      '\nlabel: {4:.3f}' \
                      '\nmax_label: {5} {6:.3f}'.format(det_idx, correct_class, det_analysis['pPDQ'],
                                                        det_analysis['spatial'], det_analysis['label'],
                                                        max_class, np.amax(det_inst.class_list))

        else:
            # Detections without full statistics state their max non-none class and confidence thereof
            max_class = class_list[np.argmax(det_inst.class_list)]
            max_score = np.amax(det_inst.class_list)
            # Note that max_class 'none' only occurs for rvc1.
            # Reading pbox json files filters out 'none' predictions in coco format as coco has no 'none' class
            # so this never happens
            if max_class == 'none':
                max_class = class_list[np.argsort(det_inst.class_list)[-2]]
                max_score = det_inst.class_list[np.argsort(det_inst.class_list)[-2]]
            det_str = '[{0}]: {1} {2:.3f}'.format(det_idx, max_class, max_score)

        ax.text(det_box[0], det_box[1], det_str, horizontalalignment='left',
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.3), fontsize=_FONTSIZE)

    # Save final image to file
    save_file = os.path.join(save_folder, os.path.splitext(os.path.basename(img_name))[0]+'.png')
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis('off')
    plt.savefig(save_file, dpi=100)
    plt.close()


def main():
    # Load all relevant information for this sequence of information
    # gt and det information
    gt_instances, det_instances, class_list = load_gt_and_det_data(args.ground_truth, args.det_json, args.data_type)

    # Analysis info
    with open(args.det_analysis, 'r') as f:
        det_analysis = json.load(f)
    with open(args.gt_analysis, 'r') as f:
        gt_analysis = json.load(f)

    # check that analysis and respective instances are the same size
    if len(gt_instances) != len(gt_analysis):
        sys.exit("ERROR! gt_instances and gt_analysis are not the same length."
                 "\ngt_instances: {0}, gt_analysis: {1}".format(len(gt_instances), len(gt_analysis)))
    if len(det_instances) != len(det_analysis):
        sys.exit("ERROR! det_instances and det_analysis are not the same length."
                 "\ndet_instances: {0}, det_analysis: {1}".format(len(det_instances), len(det_analysis)))
    
    all_images = sorted(glob.glob(os.path.join(args.gt_img_folder, '*.'+args.img_type)))
    if len(all_images) != len(det_instances):
        sys.exit("ERROR! Ground truth images (--gt_img_folder) and det_instances are not the same length."
                 "\ngt_img_folder: {0}, det_instances: {1}".format(len(all_images), len(det_instances)))

    img_data_sequence = zip(all_images,
                            gt_instances, det_instances, gt_analysis, det_analysis)
    # Go over each image and draw appropriate
    print(args.img_set)
    for img_name, img_gts, img_dets, img_gt_analysis, img_det_analysis in tqdm(img_data_sequence,
                                                                               total=len(gt_analysis),
                                                                               desc='image drawing'
                                                                               ):
        if args.img_set is None or os.path.basename(img_name) in args.img_set:
            save_analysis_img(img_name, img_gts, img_dets, img_gt_analysis, img_det_analysis, class_list,
                              args.save_folder, args.full_info, args.corner_mode)


if __name__ == '__main__':
    main()
