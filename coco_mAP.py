import numpy as np
import utils
import sys
from data_holders import PBoxDetInst, BBoxDetInst
# TODO update to be less of a hack
sys.path.append('/home/davidhall/Documents/postdoc_2018-2020/projects/rvc_new_metrics_sandbox/sandbox/'
                'evaluator_tools/metric_downloaded_code/cocoapi/PythonAPI/')
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


_HEATMAP_THRESH = 0.00135
_BLANK_IMG_SHAPE = [100, 100]

def coco_mAP(param_sequence, use_heatmap=True):
    # Create ground truth COCO object
    coco_gt = COCO()

    # Create ground truth dictionary in same format as coco expects
    # Making a lot of assumptions for default.
    # Assume class IDs currently start at 0
    # Assume image IDs currently start at 0
    # TODO implement changes to GT data holder to relate back to name of image instead of meaningless default
    # TODO implement changes to GT data holder to relate back to name of class instead of meaningless default

    coco_gt_dict = {"annotations": [], "type": "instances", "categories": [],
                    "images": []}
    coco_det_list = []

    # note that stored img_ids and stored_labels are formatted in original format (assuming start at 0)
    stored_labels = []
    # note stored annotations is formatted {<coco_image_id> : [<coco_instance_id1>, <coco_instance_id2> ...]}
    current_ann_id = 1

    # Check if there are no detections here at all. If so return a score of zero.
    # TODO return this somehow if possible
    # num_detections = np.sum([len(img_dets) for img_dets in det_instances])
    # if num_detections == 0:
    #     return 0.0

    coco_ann_ids = [[] for _ in range(len(param_sequence))]
    coco_det_ids = [[] for _ in range(len(param_sequence))]
    coco_img_ids = []
    # go through each image to create gt dict and det list of dicts
    for img_idx, (img_gt_instances, img_det_instances, _) in enumerate(param_sequence):
        # print('Img IDX: {0}'.format(img_idx))
        # if img_idx % 100 == 0:
        #     print('Img IDX: {0}'.format(img_idx))
        # Handle images with no gt instances
        if len(img_gt_instances) == 0:
            if len(img_det_instances) == 0:
                coco_img_ids += [-1]
                continue
            else:
                # Add the image to the list
                coco_gt_dict['images'].append({'id': img_idx + 1,
                                               'height': _BLANK_IMG_SHAPE[0],
                                               'file_name': '{}.jpg'.format(img_idx + 1),
                                               'width': _BLANK_IMG_SHAPE[1]})
                coco_img_ids += [img_idx + 1]
        else:
            # Add the image to the list
            coco_gt_dict['images'].append({'id': img_idx + 1,
                                           'height': img_gt_instances[0].segmentation_mask.shape[0],
                                           'file_name': '{}.jpg'.format(img_idx + 1),
                                           'width': img_gt_instances[0].segmentation_mask.shape[1]})
            coco_img_ids += [img_idx + 1]

            for gt_idx, gt_instance in enumerate(img_gt_instances):

                # Check if current instance's categoryID is currently in coco_gt_dict and if not update accordingly
                if gt_instance.class_label not in stored_labels:
                    stored_labels.append(gt_instance.class_label)
                    coco_gt_dict['categories'].append({"supercategory": "object",
                                                       "name": str(gt_instance.class_label+1),
                                                       "id": gt_instance.class_label+1})

                # Add annotation to coco_gt_dict
                coco_gt_box = gt_instance.bounding_box.copy()
                coco_gt_box[2] -= coco_gt_box[0]
                coco_gt_box[3] -= coco_gt_box[1]
                coco_gt_dict['annotations'].append({'bbox': coco_gt_box, 'iscrowd': 0,
                                                    'category_id': gt_instance.class_label+1,
                                                    'image_id': img_idx+1,
                                                    'id': current_ann_id,
                                                    'area': coco_gt_box[2]*coco_gt_box[3]})
                coco_ann_ids[img_idx] += [current_ann_id]
                current_ann_id += 1


        # Create coco detections for each detection in this image
        for det_idx, det_instance in enumerate(img_det_instances):
            # print('Det IDX: {0}'.format(det_idx))
            coco_det_class = det_instance.get_max_class()+1
            coco_det_score = det_instance.get_max_score()
            coco_det_img = img_idx+1
            if use_heatmap:
                coco_det_box = utils.generate_bounding_box_from_mask(det_instance.calc_heatmap(img_gt_instances[0].segmentation_mask.shape) > _HEATMAP_THRESH)
            else:
                if isinstance(det_instance, PBoxDetInst) or isinstance(det_instance, BBoxDetInst):
                    coco_det_box = det_instance.box.copy()
            coco_det_box[2] -= coco_det_box[0]
            coco_det_box[3] -= coco_det_box[1]

            coco_det_list.append({'bbox': coco_det_box, 'category_id': coco_det_class,
                                  'score': coco_det_score, 'image_id': coco_det_img})
            coco_det_ids[img_idx] += [len(coco_det_list)]

    for i in range(80):
        num_categories = len(coco_gt_dict['categories'])
        for j in range(num_categories):
            if i+1 == coco_gt_dict['categories'][j]['id']:
                continue

            coco_gt_dict['categories'].append({"supercategory": "object",
                                               "name": str(i+1),
                                               "id": i+1})
            break



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


    # Check that there are precisions here, if not return zero
    if len(precisions[precisions > -1]) == 0:
        return 0.0
    else:
        return np.mean(precisions[precisions > -1])
