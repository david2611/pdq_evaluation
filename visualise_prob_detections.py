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
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', choices=['coco', 'rvc1'], help='type of data being evaluated')
parser.add_argument('--gt_img_folder', help='folder with all gt images in order of gt_instances')
parser.add_argument('--det_json', help='filename for detection file to be matched with the ground-truth')
parser.add_argument('--save_folder', help='location where all analysis images will be stored')
parser.add_argument('--set_cov', type=float, help='set covariance for all det corners')
parser.add_argument('--img_type', help='type of image in gt_img_folder (png or jpg)')
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

# Define colour maps
# TODO update colour map to be at least 80 distinct colours or create distinct colour colourspace
colours = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000",
           "#800000", "#008000", "#000080", "#808000", "#800080", "#008080", "#808080",
           "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0",
           "#400000", "#004000", "#000040", "#404000", "#400040", "#004040", "#404040",
           "#200000", "#002000", "#000020", "#202000", "#200020", "#002020", "#202020",
           "#600000", "#006000", "#000060", "#606000", "#600060", "#006060", "#606060",
           "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0", "#00A0A0", "#A0A0A0",
           "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0", "#E0E0E0"
           ]

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


def load_det_data(det_json, data_type):
    """
    Function for loading ground-truth and detection instances for a single folder/sequence
    :param gt_loc: location of folder/file where ground-truth information can be extracted
    :param det_json: detection json file for the given folder/sequence
    :param data_type: string defining if we are analysing 'coco' or 'rvc1' data as this will effect
    what, and how, data is loaded given the specific filtering process of rvc1.
    :return: gt_instances, det_instances, and class list
    """
    if data_type == 'coco':
        # output is a generator of lists of DetectionInstance objects (BBox or PBox depending)
        det_instances, class_list = read_files.read_pbox_json(det_json, override_cov=args.set_cov, get_class_names=True)
    elif data_type == 'rvc1':
        # TODO update
        sys.exit("ERROR! not yet implemented")
        # gt_instances = rvc1_gt_loader.SequenceGTLoader(gt_loc)
        # det_instances = rvc1_submission_loader.DetSequenceLoader(det_json)
        # class_list = rvc1_class_list.CLASSES
    else:
        sys.exit("ERROR! Invalid data type provided")

    return det_instances, class_list


def save_analysis_img(img_name, img_dets, class_list, save_folder, corner_mode):
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

    # add detection boxes to the image
    for det_idx, det_inst in enumerate(img_dets):

        det_box = det_inst.box

        # display based on max non-none class and confidence thereof
        max_class_id = np.argmax(det_inst.class_list)
        max_class = class_list[max_class_id]
        max_score = np.amax(det_inst.class_list)
        # TODO fix to make it able to cope with synonyms in this case
        if max_class == 'none' or max_class == 'background':
            max_class_id = np.argsort(det_inst.class_list)[-2]
            max_class = class_list[max_class_id]
            max_score = det_inst.class_list[np.argsort(det_inst.class_list)[-2]]
        det_str = '[{0}]: {1} {2:.3f}'.format(det_idx, max_class, max_score)

        # TODO Choose colour based upon max class detected
        colour = colours[max_class_id % len(colours)]
        ax.add_patch(
            patches.Rectangle((det_box[0], det_box[1]), (det_box[2] - det_box[0]) + 1, (det_box[3] - det_box[1]) + 1,
                              edgecolor=colour, facecolor='none', linewidth=3))
        # draw covariances
        if isinstance(det_inst, PBoxDetInst):
            det_covs = det_inst.covs
            utils.draw_cov(det_box, det_covs, ax, colour=colour, mode=corner_mode)



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
    det_instances, class_list = load_det_data(args.det_json, args.data_type)

    img_data_sequence = zip(sorted(glob.glob(os.path.join(args.gt_img_folder, '*.'+args.img_type))),
                            det_instances)
    # Go over each image and draw appropriate
    print(args.img_set)
    for img_name, img_dets in tqdm(img_data_sequence, total=len(det_instances), desc='image drawing'):
        if args.img_set is None or os.path.basename(img_name) in args.img_set:
            save_analysis_img(img_name, img_dets, class_list, args.save_folder, args.corner_mode)


if __name__ == '__main__':
    main()
