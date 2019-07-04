import utils
from data_holders import PBoxDetInst, BBoxDetInst, ProbSegDetInst

_HEATMAP_THRESH = 0.00135
_BLANK_IMG_SHAPE = [100, 100]

def generate_coco_ground_truth_and_detections(param_sequence, use_heatmap=True):
    """
    Function for creating ground-truth dictionary and detections list in COCO format from parameter sequence of
    GroundTruthInstances and DetectionInstances.
    Note currently assumes classIDs and imageIDs start at 0
    :param param_sequence: ParamSequenceHolder containing all GroundTruthInstances and DetectionInstances
    across all sequences being evaluated.
    :param use_heatmap: Boolean dictating that BBoxes should be calculated using outskirts of heatmap rather than the
    box corner locations of PBoxDetInst or BBoxDetInst objects'
    :return:(coco_gt_dict, coco_det_list).
    coco_gt_dict: Dictionary of ground-truth annotations in COCO format
    coco_det_list: List of detections in COCO format
    """

    coco_gt_dict = {"annotations": [], "type": "instances", "categories": [],
                    "images": []}
    coco_det_list = []

    # note that stored img_ids and stored_labels are formatted in original format (assuming start at 0)
    stored_labels = []

    # note stored annotations is formatted {<coco_image_id> : [<coco_instance_id1>, <coco_instance_id2> ...]}
    current_ann_id = 1

    coco_ann_ids = [[] for _ in range(len(param_sequence))]
    coco_det_ids = [[] for _ in range(len(param_sequence))]
    coco_img_ids = []

    # go through each image to create gt dict and det list of dicts
    for img_idx, (img_gt_instances, img_det_instances) in enumerate(param_sequence):

        # Handle images with no gt instances
        if len(img_gt_instances) == 0:
            # Add blank image to the images list
            coco_gt_dict['images'].append({'id': img_idx + 1,
                                           'height': _BLANK_IMG_SHAPE[0],
                                           'file_name': '{}.jpg'.format(img_idx + 1),
                                           'width': _BLANK_IMG_SHAPE[1]})
            coco_img_ids += [img_idx + 1]
        else:
            # Add the image to the image list (invented file_name)
            coco_gt_dict['images'].append({'id': img_idx + 1,
                                           'height': img_gt_instances[0].segmentation_mask.shape[0],
                                           'file_name': '{}.jpg'.format(img_idx + 1),
                                           'width': img_gt_instances[0].segmentation_mask.shape[1]})
            coco_img_ids += [img_idx + 1]

            # Go through all ground-truth instances, adding categories where needed
            for gt_idx, gt_instance in enumerate(img_gt_instances):

                # Check if current instance's categoryID is currently in categories list and if not update accordingly
                if gt_instance.class_label not in stored_labels:
                    stored_labels.append(gt_instance.class_label)

                    # artificial supercategory and category name and id will be the same
                    coco_gt_dict['categories'].append({"supercategory": "object",
                                                       "name": str(gt_instance.class_label + 1),
                                                       "id": gt_instance.class_label + 1})

                # Add annotation to annotations list
                # Convert gt_box from rvc1 format to coco format
                coco_gt_box = [float(box_val) for box_val in gt_instance.coco_bounding_box]
                coco_gt_box[2] -= coco_gt_box[0]
                coco_gt_box[3] -= coco_gt_box[1]
                # Use the coco_iscrowd, coco_ignore, and coco_area aspects of GroundTruthInstance to ensure consistency
                coco_gt_dict['annotations'].append({'bbox': coco_gt_box,
                                                    'iscrowd': gt_instance.coco_iscrowd,
                                                    'ignore': gt_instance.coco_ignore,
                                                    'category_id': gt_instance.class_label + 1,
                                                    'image_id': img_idx + 1,
                                                    'id': current_ann_id,
                                                    'area': gt_instance.coco_area})
                coco_ann_ids[img_idx] += [current_ann_id]
                current_ann_id += 1

        # Create coco detections for each detection in this image
        for det_idx, det_instance in enumerate(img_det_instances):
            if isinstance(det_instance, ProbSegDetInst):
                coco_det_class = det_instance.chosen_label
            else:
                coco_det_class = int(det_instance.get_max_class() + 1)
            coco_det_score = float(det_instance.get_max_score())
            coco_det_img = img_idx + 1

            # If using heatmap, generate a bounding box that fits the heatmap
            if use_heatmap:
                coco_det_box = utils.generate_bounding_box_from_mask(
                    det_instance.calc_heatmap(img_gt_instances[0].segmentation_mask.shape) > _HEATMAP_THRESH)
            else:
                if isinstance(det_instance, (PBoxDetInst, BBoxDetInst, ProbSegDetInst)):
                    coco_det_box = [float(boxval) for boxval in det_instance.box]
                else:
                    raise ValueError("Cannot create bbox for detection! Not using heatmap, PBoxDetInst, or BBoxDetInst")

            # Convert the box from rvc1 format to coco format
            coco_det_box[2] -= coco_det_box[0]
            coco_det_box[3] -= coco_det_box[1]

            coco_det_list.append({'bbox': coco_det_box, 'category_id': coco_det_class,
                                  'score': coco_det_score, 'image_id': coco_det_img})
            coco_det_ids[img_idx] += [len(coco_det_list)]

    # Assuming that 80 classes are expected for COCO calculations, add categories if we have less than 80.
    for i in range(80):
        num_categories = len(coco_gt_dict['categories'])
        need_new_cat = True
        for j in range(num_categories):
            if i + 1 == coco_gt_dict['categories'][j]['id']:
                need_new_cat = False
                break

        if need_new_cat:
            print("Missed a category. Appending")
            coco_gt_dict['categories'].append({"supercategory": "object",
                                               "name": str(i + 1),
                                               "id": i + 1})

    return coco_gt_dict, coco_det_list