"""
Any time COCO data is read, we convert to GroundTruthInstances and DetectionInstances.
We require official COCO code to be downloaded and installed. Link to code: https://github.com/cocodataset/cocoapi
System path must be appended to include location of PythonAPI.
"""

import numpy as np
from data_holders import GroundTruthInstance, PBoxDetInst, BBoxDetInst, ProbSegDetInst
import json
import time
from itertools import islice
import sys
import os.path as osp

# Temp way to get access to COCO code for now
sys.path.append('/path/to/COCO/PythonAPI/')
from pycocotools.coco import COCO


def read_pbox_json(filename, gt_class_ids=None, get_img_names=False, get_class_names=False, n_imgs=None,
                   override_cov=None, label_threshold=0, prob_seg=False):
    """
    The following function reads a json file design to describe detections which can be expressed as
    probabilistic bounding boxes and return a list of list of detection instances for each image,
    the list of image names associated with the sequence shown by the .json file,
    and the list of class names which correspond to the label_probs for each detection.
    :param filename: filename where the desired json file is stored (.json included)
    :param gt_class_ids: Dictionary with class names as keys and class idxs as values. Used for ordering detections into
    format of ground-truth (Default None which assumes order is already accurate)
    :param get_img_names: flag to determine if we expect function to return the image names in order (default False)
    :param get_class_names: flag to determine if we expect function to return the class names in order (default False)
    :param n_imgs: allows choosing the number of images we want to read detections for (default None meaning all images)
    :param override_cov: Override the reported covariance while loading, use for calibration.
    If zero, BBox detections are used
    :param label_threshold: label threshold for maximum non-background class for detection to be loaded
    (default zero meaning all detections loaded)
    :param prob_seg: flag determining if detections are of probabilistic segmentation format
    :return:
    detection_instances: list of list of detection instances for each image
    img_names: list of image names for each image (if flagged as desired by get_img_names)
    class_names: list of class names for each class expressed within label_probs of each detection
    (if flagged as desired by get_class_names)
    """
    time_start = time.time()

    # Read data from json file
    with open(filename, 'r') as f:
        data_dict = json.load(f)

    # Associate detection classes to ground truth classes, to get the order right (for evaluation)
    # format: class_association[det_idx] = gt_idx
    class_association = {}
    for det_idx, det_class in enumerate(data_dict['classes']):
        # If no ground-truth classes available, assume order is already right
        # Note this should mostly be used only for visualisation of raw detections
        if gt_class_ids is None:
            class_association[det_idx] = det_idx
        else:
            # Find the gt_class in the gt_class_ids dictionary
            for gt_class in gt_class_ids.keys():
                if are_classes_same(det_class, gt_class):
                    class_association[det_idx] = gt_class_ids[gt_class]
                    break

    det_ids = sorted(class_association.keys())
    gt_ids = [class_association[idx] for idx in det_ids]

    # extract the maximum number of classes
    if gt_class_ids is None:
        num_classes = len(data_dict['classes'])
    else:
        num_classes = max(class_id for class_id in gt_class_ids.values()) + 1

    # create a detection instance for each detection described by dictionaries in dict_dets
    if prob_seg:
        det_instances = ProbSegmentLoader(data_dict['detections'], (gt_ids, det_ids, num_classes), filename, n_imgs, label_threshold)
    else:
        det_instances = BoxLoader(data_dict['detections'], (gt_ids, det_ids, num_classes), n_imgs,
                                  override_cov, label_threshold)
    time_end = time.time()
    print("Total Time: {}".format(time_end-time_start))

    # Return appropriate output
    if get_img_names:
        if get_class_names:
            return det_instances, data_dict['img_names'], data_dict['classes']
        else:
            return det_instances, data_dict['img_names']
    if get_class_names:
        return det_instances, data_dict['classes']
    return det_instances


class BoxLoader:

    def __init__(self, dict_dets, class_assoc, n_imgs=None, override_cov=None, label_threshold=0):
        """
        Initialiser for PBoxLoader object which reads in PBoxDetInst detections from dictionary detection information
        :param dict_dets: dictionary of detection information
        :param class_assoc: class association dictionary
        :param n_imgs: number of images loading detections for out of all available for a sequence
        :param override_cov: set covariance used to make spherical Gaussian corners (above zero)
        :param label_threshold: label threshold to apply to detections when determining whether to keep them
        """
        self.dict_dets = dict_dets
        self.class_assoc = class_assoc
        self.n_imgs = n_imgs
        self.cov_mat = None
        self.label_threshold = float(label_threshold)
        self.override_cov = override_cov
        if override_cov is not None and override_cov > 0:
            self.cov_mat = [[[override_cov, 0], [0, override_cov]], [[override_cov, 0], [0, override_cov]]]

    def __len__(self):
        return len(self.dict_dets) if self.n_imgs is None else self.n_imgs

    def __iter__(self):
        if self.n_imgs is not None:
            dets_iter = islice(self.dict_dets, 0, self.n_imgs)
        else:
            dets_iter = iter(self.dict_dets)

        for img_id, img_dets in enumerate(dets_iter):

            yield [
                # If the detection has a reported covariance above zero, create a Probabilistic Box
                PBoxDetInst(
                    class_list=reorder_classes(det['label_probs'], self.class_assoc),
                    # Removed clamping boxes for detection output standardization for now.
                    # box=clamp_bbox(det['bbox'], det['img_size']),
                    box=det['bbox'],
                    covs=det['covars'] if self.cov_mat is None else self.cov_mat
                )
                if (self.cov_mat is not None or ('covars' in det and (np.sum(det['covars']) != 0)))
                and self.override_cov != 0

                # Otherwise create a standard bounding box
                else
                BBoxDetInst(
                    class_list=reorder_classes(det['label_probs'], self.class_assoc),
                    box=det['bbox']
                )
                for det in img_dets

                # Regardless of type, ignore detections below given label threshold if provided
                # Note that this includes label_prob of background classes
                if self.label_threshold <= 0 or max(det['label_probs']) > self.label_threshold
            ]


class ProbSegmentLoader:
    def __init__(self, dict_dets, class_assoc, det_filename, n_imgs=None, label_threshold=0):
        """
        Initialiser for MaskRCNNLoader object which reads in MaskRCNNDetInst detections from dictionary detection
        information
        :param dict_dets: dictionary of detection information
        :param class_assoc: class association dictionary
        :param det_filename: filename used for loading detections (needed for determining path to mask imgs)
        :param n_imgs: number of images loading detections for out of all available for a sequence
        :param label_threshold: label threshold to apply to detections when determining whether to keep them
        """
        self.dict_dets = dict_dets
        self.class_assoc = class_assoc
        self.n_imgs = n_imgs
        self.label_threshold = float(label_threshold)
        self.det_filename = det_filename

    def __len__(self):
        return len(self.dict_dets) if self.n_imgs is None else self.n_imgs

    def __iter__(self):
        if self.n_imgs is not None:
            dets_iter = islice(self.dict_dets, start=0, stop=self.n_imgs)
        else:
            dets_iter = iter(self.dict_dets)
        for img_id, img_dets in enumerate(dets_iter):
            yield [
                ProbSegDetInst(
                    class_list=reorder_classes(det['label_probs'], self.class_assoc),
                    chosen_label=None if 'label' not in det else det['label'],
                    mask_id=det['mask_id'],
                    detection_file=det['masks_file'],
                    # Only provide a mask root if path given is not absolute
                    mask_root='' if osp.isabs(det['masks_file']) else osp.dirname(osp.abspath(self.det_filename)),
                    box=None if 'bbox' not in det else det['bbox']
                )
                for det in img_dets
                # Regardless of type, ignore detections below given label threshold if provided
                # Note that this includes label_prob of background classes
                if self.label_threshold <= 0 or max(det['label_probs']) > self.label_threshold
            ]


def read_COCO_gt(filename, n_imgs=None, ret_img_sizes=False, ret_classes=False, bbox_gt=False):
    """
    Function for reading COCO ground-truth files and converting them to GroundTruthInstances format.
    :param filename: filename of the annotation.json file with all COCO ground-truth annotations
    :param n_imgs: number of images ground-truth is being extracted from. If None extract all (default None)
    :param ret_img_sizes: Boolean flag dictating if the image sizes should be returned
    :param ret_classes: Boolean flag dictating if the class mapping dictionary should be returned
    :param bbox_gt: Boolean flag dictating if the GroundTruthInstance should ignore the segmentation mask and only use
    bounding box information.
    :return: ground-truth instances as GTLoader and optionally image sizes or class mapping dictionary if requested
    """

    # read the json file
    coco_obj = COCO(filename)

    gt_instances = GTLoader(coco_obj, n_imgs, bbox_gt=bbox_gt)

    # Return image sizes if requested
    if ret_img_sizes:
        return gt_instances, [
            [coco_obj.imgs[img_id]['height'], coco_obj.imgs[img_id]['width']]
            for img_id in sorted(coco_obj.imgs.keys())
        ]

    # Return class mapping dictionary if requested
    if ret_classes:
        return gt_instances, {
            coco_obj.cats[cat_id]['name']: idx
            for idx, cat_id in enumerate(sorted(coco_obj.cats.keys()))
        }
    return gt_instances


class GTLoader:

    def __init__(self, coco_obj, n_imgs, bbox_gt=False):
        """
        Initialisation function for GTLoader object which loads ground-truth annotations from COCO and
        produces GroundTruthInstance objects.
        :param coco_obj: ground-truth object for all images in COCO annotation format
        :param n_imgs: number of images ground-truth is being extracted from. If None extract all
        """
        self.coco_obj = coco_obj
        self.n_imgs = n_imgs
        self.bbox_gt = bbox_gt

    def __len__(self):
        return len(self.coco_obj.imgs) if self.n_imgs is None else self.n_imgs

    def __iter__(self):
        coco_annotations = self.coco_obj.imgToAnns
        img_ids = sorted(self.coco_obj.imgs.keys())

        # Create map to transfer from category id to index id (used as class id in our tests)
        ann_idx_map = {
            cat_id: idx
            for idx, cat_id in enumerate(sorted(self.coco_obj.cats.keys()))
        }

        if self.n_imgs is not None:
            img_id_iter = islice(img_ids, 0, self.n_imgs)
        else:
            img_id_iter = iter(img_ids)
        for img_idx, img_id in enumerate(img_id_iter):
            if img_id not in coco_annotations.keys():
                yield []
            else:
                # load the annotations available for the given image
                # filter out any annotations which do not have segmentations
                for annotation in coco_annotations[img_id]:
                    if "segmentation" not in annotation.keys():
                        print("SKIPPED A GT OBJECT!")
                img_annotations = [
                    annotation
                    for annotation in coco_annotations[img_id]
                    if 'segmentation' in annotation.keys()
                ]
                # extract the class ids for each annotation (note that we subtract 1 so that class ids start at 0)
                class_ids = [annotation['category_id'] for annotation in img_annotations]
                bboxes = []
                seg_masks = []
                ignores = [annotation['ignore'] if 'ignore' in annotation.keys() else False
                           for annotation in img_annotations]
                iscrowds = [annotation['iscrowd'] for annotation in img_annotations]
                areas = [annotation['area'] for annotation in img_annotations]

                for annotation in img_annotations:
                    # transform bbox to [x1, y1, x2, y2]
                    box = annotation['bbox']
                    box[2] += box[0]
                    box[3] += box[1]
                    bboxes.append(box)

                    # define GT segmentation mask
                    # If segmentation mask is expected to be pixels within bounding box, adjust accordingly
                    seg_mask = self.coco_obj.annToMask(annotation)
                    if self.bbox_gt:
                        eval_mask = np.zeros(seg_mask.shape, dtype=np.bool)

                        # Use the COCO bounding box (note not exact around segmentation masks)
                        # Round down for lower bounds and round up for upper bounds to accommodate floats
                        seg_bbox = np.floor(box).astype(np.int)
                        seg_bbox[-2:] = np.ceil(box[-2:]).astype(np.int)

                        # clamp box to image size to make sure not outside extremes
                        seg_bbox = clamp_bbox(seg_bbox, eval_mask.shape)

                        # Create segmentation mask with bounding box
                        eval_mask[seg_bbox[1]:seg_bbox[3]+1, seg_bbox[0]:seg_bbox[2]+1] = True

                        # Below use seg_bbox for simplicity rather than trying to account for rounding in box
                        # (USED IN REBUTTAL)
                        # seg_bbox = generate_bounding_box_from_mask(seg_mask)
                        # eval_mask[seg_bbox[1]:seg_bbox[3]+1, seg_bbox[0]:seg_bbox[2]+1] = True

                        seg_masks.append(eval_mask)
                    else:
                        seg_masks.append(seg_mask)

                # generate ground truth instances from the COCO annotation information
                # NOTE this will skip any annotation which has a bad segmentation mask (all zeros)
                yield [
                    GroundTruthInstance(
                        segmentation_mask=seg_masks[ann_idx],
                        true_class_label=ann_idx_map[class_ids[ann_idx]],
                        coco_bounding_box=bboxes[ann_idx],
                        coco_ignore=ignores[ann_idx],
                        coco_iscrowd=iscrowds[ann_idx],
                        coco_area=areas[ann_idx]
                        )
                    for ann_idx in range(len(img_annotations)) if np.amax(seg_masks[ann_idx] > 0)
                ]


def convert_coco_det_to_rvc_det(det_filename, gt_filename, save_filename):
    """
    Function for converting COCO format detection file into RVC1 format detection file
    :param det_filename: filename for original detections in COCO format
    :param gt_filename: filename for ground-truth in COCO format
    :param save_filename: filename where detections in RVC1 format will be saved
    :return: None
    """
    coco_obj = COCO(gt_filename)

    with open(det_filename, 'r') as fp:
        det_coco_dicts = json.load(fp)

    # Extract primary information
    gt_img_ids = sorted(coco_obj.imgs.keys())
    det_img_ids = np.array([det_dict['image_id'] for det_dict in det_coco_dicts])
    rvc1_dets = []
    class_list = [coco_obj.cats[class_id]['name'] for class_id in sorted(coco_obj.cats.keys())]
    ann_idx_map = {
        cat_id: idx
        for idx, cat_id in enumerate(sorted(coco_obj.cats.keys()))
    }

    # Assume covariances will be zero (BBox detections)
    empty_covars = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]

    # Go through all images in coco gt and make det inputs for them
    for img_idx, img_id in enumerate(gt_img_ids):
        img_coco_dets = [det_coco_dicts[idx] for idx in np.argwhere(det_img_ids == img_id).flatten()]
        img_rvc1_dets = []
        for det_idx, det in enumerate(img_coco_dets):
            # Note, currently assumes the detection has a bbox entry.
            # Transform to [x1, y1, x2, y2] format
            box = det['bbox']
            box[2] += box[0]
            box[3] += box[1]

            # Checking if the json has all the predicted scores
            if 'all_scores' in det:
                label_probs = det['all_scores']
                if len(label_probs) != len(class_list):
                    sys.exit(f'ERROR! "all_scores" array for image {img_id} has size {len(label_probs)} but should have size {len(class_list)}')
            else:
                # Extract score of chosen class and distribute remaining probability across all others
                label_probs = np.ones(len(class_list)) * ((1 - det['score'])/(len(class_list)-1))
                label_probs[ann_idx_map[det['category_id']]] = det['score']
                label_probs = list(label_probs.astype(float))
            
            # Checking if the json has pre-calculated covariance matrices
            if 'covar_xyxy' in det and det['covar_xyxy'] is not None:
                covars_vals = det['covar_xyxy']
                covars_shape = np.array(covars_vals).shape
                if covars_shape != (2, 2, 2):
                    sys.exit(f'ERROR! "covar_xyxy" matrix for image {img_id} has shape {covars_shape} but should have shape (2, 2, 2)')
            else:
                covars_vals = empty_covars
            
            det_dict = {'bbox': box, 'covars': covars_vals, "label_probs": label_probs}
            img_rvc1_dets.append(det_dict)

        rvc1_dets.append(img_rvc1_dets)
    save_dict = {'classes': class_list, "detections": rvc1_dets}

    # Save detections in rvc1 format
    with open(save_filename, 'w') as f:
        json.dump(save_dict, f)


def are_classes_same(class_1, class_2):
    """
    Synonym handling
    :param class_1: First class name string
    :param class_2: Second class name string
    :return: Boolean dictating if two class name strings are equivalent
    """
    class_1 = class_1.lower()
    class_2 = class_2.lower()
    if class_1 == class_2:
        return True
    for synset in [
        {'background', '__background__', '__bg__', 'none'},
        {'motorcycle', 'motorbike'},
        {'aeroplane', 'airplane'},
        {'traffic light', 'trafficlight'},
        {'sofa', 'couch'},
        {'pottedplant', 'potted plant'},
        {'diningtable', 'dining table'},
        {'stop sign', 'stopsign'},
        {'tvmonitor', 'tv', 'television', 'computer monitor'}
    ]:
        if class_1 in synset and class_2 in synset:
            return True
        elif class_1 in synset or class_2 in synset:
            # Each class should only ever be in 1 sysnset, fail out if we found only one
            return False
    return False


def reorder_classes(label_probs, class_assoc):
    class_probs = np.zeros(class_assoc[2], dtype=np.float32)
    class_probs[class_assoc[0]] = np.array(label_probs)[class_assoc[1]]
    return class_probs


def clamp_bbox(box, image_size):
    return [
        max(box[0], 0),
        max(box[1], 0),
        min(image_size[1], box[2]),
        min(image_size[0], box[3]),
    ]
