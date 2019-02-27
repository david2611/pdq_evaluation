import numpy as np
from data_holders import GroundTruthInstance, PBoxDetInst, BBoxDetInst
import json
import time
from itertools import islice
import sys
# Temp way to get access to COCO code for now
sys.path.append('/home/davidhall/Documents/postdoc_2018-2020/projects/rvc_new_metrics_sandbox/sandbox/'
           'evaluator_tools/metric_downloaded_code/cocoapi/PythonAPI/')
from pycocotools.coco import COCO


def read_pbox_json(filename, gt_class_ids, get_img_names=False, get_class_names=False, n_imgs=None,
                   override_cov=None, label_threshold=0):
    """
    The following function reads a json file design to describe detections which can be expressed as
    probabilistic bounding boxes and return a list of list of detection instances for each image,
    the list of image names associated with the sequence shown by the .json file,
    and the list of class names which correspond to the label_probs for each detection.
    :param filename: filename where the desired json file is stored (.json included)
    :param get_img_names: flag to determine if we expect this function to return the image names in order (default False)
    :param get_class_names: flag to determine if we expect this function to return the class names in order (default False)
    :param override_cov: Override the reported covariance while loading, use for calibration
    :return:
    detection_instances: list of list of detection instances for each image
    img_names: list of image names for each image (if flagged as desired by get_img_names)
    class_names: list of class names for each class expressed within label_probs of each detection (if flagged as desired by get_class_names)
    """
    time_start = time.time()

    # Read data from json file
    with open(filename, 'r') as f:
        data_dict = json.load(f)

    # Associate detection classes to ground truth classes, to get the order right
    class_association = {}
    for det_idx, det_class in enumerate(data_dict['classes']):
        for gt_class in gt_class_ids.keys():
            if are_classes_same(det_class, gt_class):
                class_association[det_idx] = gt_class_ids[gt_class]
                break
    det_ids = sorted(class_association.keys())
    gt_ids = [class_association[idx] for idx in det_ids]
    num_classes = max(class_id for class_id in gt_class_ids.values()) + 1

    # TODO made funcitonality for changing number of image's detections are loaded but not checked all implications
    # create a detection instance for each detection described by dictionaries in dict_dets
    if override_cov == 0:
        det_instances = BBoxLoader(data_dict['detections'], (gt_ids, det_ids, num_classes), n_imgs, label_threshold)
    else:
        det_instances = PBoxLoader(data_dict['detections'], (gt_ids, det_ids, num_classes), n_imgs,
                                   override_cov, label_threshold)
    time_end = time.time()
    print("Total Time: {}".format(time_end-time_start))
    if get_img_names:
        if get_class_names:
            return det_instances, data_dict['img_names'], data_dict['classes']
        else:
            return det_instances, data_dict['img_names']
    if get_class_names:
        return det_instances, data_dict['classes']
    return det_instances


class BBoxLoader:

    def __init__(self, dict_dets, class_assoc, n_imgs=None, label_threshold=0):
        self.dict_dets = dict_dets
        self.class_assoc = class_assoc
        self.n_imgs = n_imgs
        self.label_threshold = float(label_threshold)

    def __len__(self):
        return len(self.dict_dets) if self.n_imgs is None else self.n_imgs

    def __iter__(self):
        if self.n_imgs is not None:
            dets_iter = islice(self.dict_dets, start=0, stop=self.n_imgs)
        else:
            dets_iter = iter(self.dict_dets)
        for img_id, img_dets in enumerate(dets_iter):
            yield [
                BBoxDetInst(
                    class_list=reorder_classes(det['label_probs'], self.class_assoc),
                    # Removed clamping boxes for detection output standardization for now.
                    # box=clamp_bbox(det['bbox'], det['img_size'])
                    box=det['bbox']
                )
                for det in img_dets
                if self.label_threshold <= 0 or max(det['label_probs']) > self.label_threshold
            ]


class PBoxLoader:

    def __init__(self, dict_dets, class_assoc, n_imgs=None, override_cov=None, label_threshold=0):
        self.dict_dets = dict_dets
        self.class_assoc = class_assoc
        self.n_imgs = n_imgs
        self.cov_mat = None
        self.label_threshold = float(label_threshold)
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
            yield [PBoxDetInst(
                class_list=reorder_classes(det['label_probs'], self.class_assoc),
                # Removed clamping boxes for detection output standardization for now.
                # box=clamp_bbox(det['bbox'], det['img_size']),
                box=det['bbox'],
                covs=det['covars'] if self.cov_mat is None else self.cov_mat
            )
                for det in img_dets
                if self.label_threshold <= 0 or max(det['label_probs']) > self.label_threshold
            ]


def patch_image_size(coco_gt, detections_file):
    coco_obj = COCO(coco_gt)

    with open(detections_file, 'r') as fp:
        data_dict = json.load(fp)

    img_ids = sorted(coco_obj.imgs.keys())
    for img_idx, img_id in enumerate(img_ids):
        for det in data_dict['detections'][img_idx]:
            det['img_size'] = [coco_obj.imgs[img_id]['height'], coco_obj.imgs[img_id]['width']]

    with open(detections_file, 'w') as fp:
        json.dump(data_dict, fp)


def read_COCO_gt(filename, n_imgs=None, ret_img_sizes=False, ret_classes=False):

    # read the json file
    coco_obj = COCO(filename)

    # Create location for gt_instances for each image to be stored
    gt_instances = GTLoader(coco_obj, n_imgs)
    if ret_img_sizes:
        return gt_instances, [
            [coco_obj.imgs[img_id]['height'], coco_obj.imgs[img_id]['width']]
            for img_id in sorted(coco_obj.imgs.keys())
        ]
    if ret_classes:
        return gt_instances, {
            coco_obj.cats[cat_id]['name']: idx
            for idx, cat_id in enumerate(sorted(coco_obj.cats.keys()))
        }
    return gt_instances


class GTLoader:

    def __init__(self, coco_obj, n_imgs):
        self.coco_obj = coco_obj
        self.n_imgs = n_imgs

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
                img_annotations = [
                    annotation
                    for annotation in coco_annotations[img_id]
                    if 'segmentation' in annotation.keys()
                ]
                # extract segmentation masks for each annotation
                seg_masks = [self.coco_obj.annToMask(annotation) for annotation in img_annotations]
                # extract the class ids for each annotation (note that we subtract 1 so that class ids start at 0)
                class_ids = [annotation['category_id'] for annotation in img_annotations]
                # generate ground truth instances from the COCO annotation information
                # NOTE this will skip any annotation which has a bad segmentation mask (all zeros)
                yield [
                    GroundTruthInstance(
                        segmentation_mask=seg_masks[ann_idx],
                        true_class_label=ann_idx_map[class_ids[ann_idx]],
                        )
                    for ann_idx in range(len(img_annotations)) if np.amax(seg_masks[ann_idx] > 0)
                ]


def are_classes_same(class_1, class_2):
    """
    Synonym handling
    :param class_1:
    :param class_2:
    :return:
    """
    class_1 = class_1.lower()
    class_2 = class_2.lower()
    if class_1 == class_2:
        return True
    for synset in [
        {'background', '__background__', '__bg__'},
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
