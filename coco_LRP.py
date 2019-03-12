import numpy as np
import utils
import sys
from data_holders import PBoxDetInst, BBoxDetInst
# TODO update to be less of a hack
# Temp way to get access to COCO code for now
sys.path.append('/mnt/storage_device/postdoc_2018-2020/projects/LRP_evaluation/LRP/cocoLRPapi-master/PythonAPI/')
from pycocotools_lrp.cocoevalLRP import COCOevalLRP
from pycocotools_lrp.coco import COCO


_HEATMAP_THRESH = 0.00135
_BLANK_IMG_SHAPE = [100, 100]

def coco_LRP(param_sequence, use_heatmap=True, full=False):
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
            # Add blank image to the list
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
                coco_gt_box = [float(box_val) for box_val in gt_instance.coco_bounding_box]
                coco_gt_box[2] -= coco_gt_box[0]
                coco_gt_box[3] -= coco_gt_box[1]
                coco_gt_dict['annotations'].append({'bbox': coco_gt_box,
                                                    'iscrowd': gt_instance.coco_iscrowd,
                                                    'ignore': gt_instance.coco_ignore,
                                                    'category_id': gt_instance.class_label+1,
                                                    'image_id': img_idx+1,
                                                    'id': current_ann_id,
                                                    'area': gt_instance.coco_area})
                coco_ann_ids[img_idx] += [current_ann_id]
                current_ann_id += 1


        # Create coco detections for each detection in this image
        for det_idx, det_instance in enumerate(img_det_instances):
            # print('Det IDX: {0}'.format(det_idx))
            coco_det_class = int(det_instance.get_max_class()+1)
            coco_det_score = float(det_instance.get_max_score())
            coco_det_img = img_idx+1
            if use_heatmap:
                coco_det_box = utils.generate_bounding_box_from_mask(det_instance.calc_heatmap(img_gt_instances[0].segmentation_mask.shape) > _HEATMAP_THRESH)
            else:
                if isinstance(det_instance, PBoxDetInst) or isinstance(det_instance, BBoxDetInst):
                    coco_det_box = [float(boxval) for boxval in det_instance.box]
            coco_det_box[2] -= coco_det_box[0]
            coco_det_box[3] -= coco_det_box[1]

            coco_det_list.append({'bbox': coco_det_box, 'category_id': coco_det_class,
                                  'score': coco_det_score, 'image_id': coco_det_img})
            coco_det_ids[img_idx] += [len(coco_det_list)]

    for i in range(80):
        num_categories = len(coco_gt_dict['categories'])
        need_new_cat = True
        for j in range(num_categories):
            if i+1 == coco_gt_dict['categories'][j]['id']:
                need_new_cat = False
                break

        if need_new_cat:
            print("Missed a category. Appending")
            coco_gt_dict['categories'].append({"supercategory": "object",
                                               "name": str(i+1),
                                               "id": i+1})


    # print('length of gt images', len(coco_gt_dict['images']))
    # print('length of gt categories', len(coco_gt_dict['categories']))
    # print('length of gt annotations', len(coco_gt_dict['annotations']))

    # #################TEMP CODE####################
    # import json
    # print('saving det file')
    # with open('/home/davidhall/Desktop/temp_mAP_stuff/homemade_detfile.json', 'w') as f:
    #     json.dump(coco_det_list, f)
    # print('saving gt file')
    # with open('/home/davidhall/Desktop/temp_mAP_stuff/homemade_gtfile.json', 'w') as f:
    #     json.dump(coco_gt_dict, f)

    # Finish creating the coco ground truth object
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

    if not full:
        return coco_eval_lrp.eval['moLRP']
    else:
        return {'moLRP': coco_eval_lrp.eval['moLRP'],
                'moLRPLoc': coco_eval_lrp.eval['moLRPLoc'], 'moLRPFP': coco_eval_lrp.eval['moLRPFP'],
                'moLRPFN': coco_eval_lrp.eval['moLRPFN']}

