# PDQ Evaluation

This is a fork with some small changes from the original one, also assuming `cocoapi` is a sibling folder. 


# Original README

Overview
========
This code enables the evaluation of detectors on Common Objects in Context (*COCO*) and 
Robotic Vision Challenge 1 (*RVC1*) data. Evaluation is performed using mean average precision (*mAP*) as implemented in
the COCO challenge, mean optimal localisation recall precision (*moLRP*) based metrics, 
and probabilistic detection quality (*PDQ*) as implemented (mostly) in the RVC1 CVPR2019 challenge.

The code here, particularly for evaluating RVC1 results is based heavily on the RVC1 challenge code which can be found 
here: https://github.com/jskinn/rvchallenge-evaluation

For further details on the robotic vision challenges please see the following links for more details:

Robotic Vision Challenges Homepage: http://roboticvisionchallenge.org/ 
CVPR 2019 Challenge Page: https://competitions.codalab.org/competitions/20940
Continuous Probabilistc Object Detection Challenge Page: https://competitions.codalab.org/competitions/21727

Setup
=====

Install all python requirements
------------------------
This code comes with a requirements.txt file.
Make sure you have installed all libraries as part of your working environment.

Install COCO mAP API
--------------------
After installing all requirements, you will need to have a fully installed implementation of the COCO API located
somewhere on your machine.
You can download this API here https://github.com/cocodataset/cocoapi.

Once this is downloaded and installed, you need to adjust the system path on line 11 of coco_mAP.py and line 16 of 
read_files.py to match the PythonAPI folder of your COCO API installation.

Add LRP Evaluation Code
-----------------------
You will also require code for using LRP evaluation measures.
To do this you need to simply copy the cocoevalLRP.py file from the LRP github repository to the pycocotools folder within the PythonAPI.
You can download the specific file here https://github.com/cancam/LRP/blob/master/cocoLRPapi-master/PythonAPI/pycocotools/cocoevalLRP.py
You can clone the original repository here https://github.com/cancam/LRP.

After cocoevalLRP.py is located in your pycocotools folder, simply adjust the system path on line 11 of coco_LRP.py to match your PythonAPI folder.

Usage
=====
All evaluation code is run on detections saved in .json files formatted as required by the RVC outlined later on.
A variation to this is also available for probabilistic segmentation format also described later.
If you are evaluating on COCO data and have saved detections in COCO format, you can convert to RVC1 format using 
*file_convert-coco_to_rvc1.py*
When you have the appropriate files, you can evaluate on mAP, moLRP, and PDQ with *evaluate.py*.
After evaluation is complete, you can visualise your detections for a sequence of images w.r.t. PDQ using 
*visualise_pdq_analysis.py*

Evaluation is currently organised so that you can evaluate either on COCO data, or on RVC1 data. Note that RVC1 data
expects multiple sequences rather than a single folder of data.

RVC1 Detection Format
---------------------
RVC1 detections are saved in a single .json file per sequence being evaluated. Each .json file is formatted as follows:

```
{
  "classes": [<an ordered list of class names>],
  "detections": [
    [
      {
        "bbox": [x1, y1, x2, y2],
        "covars": [
          [[xx1, xy1],[xy1, yy1]],
          [[xx2, xy2],[xy2, yy2]]
        ],
        "label_probs": [<an ordered list of probabilities for each class>]
      },
      {
      }
    ],
    [],
    []
    ...
  ]
}
```

### Probabilistic Segmentation Detections ###
We now accommodate a way to submit probabilistic segmentation detections.
For this format, a .npy file for each image stores all detection probabilistic segmentation heatmaps for that image.
This 3D array's shape is m x h x w where m is the number of segmentation masks, h is the image height, and w is the
image width. 
Each detection dictionary now contains the location the .npy file associated with the detection and the mask id for the
specific detection.
You may also define a bounding box to replace the probabilistic segmentation for bounding-box detections and define a
chosen class to use for mAP and moLRP evaluation (rather than always using max class of label_probs).

Expected format for probabilistic segmentation detection files is as follows:

```
{
  "classes": [<an ordered list of class names>],
  "detections": [
    [
      {
        "label_probs": [<an ordered list of probabilities for each class>],
        "masks_file": "<location of .npy file holding probabilistic segmentation mask>",
        "mask_id": <index of this detection's mask in mask_file's numpy array>,
        "label": <chosen label within label_probs> (optional),
        "bbox": [x1, y1, x2, y2] (optional for use in mAP and moLRP),
      },
      {
      }
    ],
    [],
    []
    ...
  ]
}
```

file_convert_coco_to_rvc1.py
----------------------------
 To convert coco detections to rvc format simply run:
 
 `python file_convert_coco_to_rvc1.py --coco_gt <gt_json_file> --coco_det <det_json_file> --rvc1_det <output_json_file>`
 
 where `<gt_json_file>` is the coco format ground-truth json filename, `det_json_file` is the coco format detection 
 json filename, and `output_file` is the json filename you will save your rvc1 formatted detections json file.
 
 ### Important Notes ###
 
 By default, coco json format does not come with the predicted scores for all the classes available, in which case the conversion script will just
 extract the score of the chosen class and distribute remaining probability across all others classes. However, this will produce
 incorrect measures of label quality because it is the probability estimated by the detector for the object's ground-truth class, which might not
 correspond to the chosen class. To facilitate correct measurements, if a detection element in the coco json file (`det_json_file`) comes with a 
 key `all_scores`, the conversion script will consider it as an array of all the scores, and use it instead of the default behaviour.
 
 Also, by default, coco json format does not consider the existence of a covariance matrix which is needed for PDQ calculations. The conversion
 script assigns by default a zero'ed covariance matrix, but if a detection element in the coco json file (`det_json_file`) comes with a 
 key `covar_xyxy`, the conversion script will use that covariance matrix instead of the default one with zeros.
 
evaluate.py
-----------
 To perform full evaluation simply run:
 
 `python evaluate.py --test_set <test_type> --gt_loc <gt_location> --det_loc <det_location> --save_folder <save_folder> --set_cov <cov>`
 
 Optional flags for new functionality include `--bbox_gt`, `--segment_mode`, `--greedy_mode`, and `--prob_seg`.
 There is also an `--mAP_heatmap` flag but that should not generally be used.
 
 - `<test_type>` is a string defining whether we are evaluating COCO or RVC1 data. Options are 'coco' and 'rvc1'

 - `<gt_location>` is a string defining either the location of a ground-truth .json file (coco tests) or a folder of
 ground truth sequences (rvc1 data). Which one it is interpreted as is defined by `<test_type>`
 
 - `<det_loc>` is a string defining either the location of a detection .json file (coco data) or a folder of .json files for 
 multiple sequences (rvc1 data). Which one it is interpreted as is defined by `<test_type>`.
 Note that these detection files must be in rvc1 format.
 
 - `<save_folder>` is a string defining the folder where analysis will be stored in form of scores.txt, and files for visualisations
 - `<cov>` is an optional value defining set covariance for the corners of detections.
 
 - `--bbox_gt` flag states that all ground-truth should be teated as bounding boxes for PDQ analysis.
 All pixels within the bounding box will be used for analysis and there will be no "ignored" pixels. This enables
 use of datasets with no segmentation information provided they are stored in COCO ground-truth format.
 
 - `--segment_mode` flag states that evaluation is performed per-pixel on the ground-truth segments with no "ignored"
 pixels to accommodate box-shaped detections. This should only be used if evaluating a probabilistic segmentation
 detection system.
 
 - `--greedy_mode` flag states that assignment of detections to ground-truth objects based upon pPDQ scores is done
 greedily rather than optimal assignment. Greedy mode can be faster for some applications but does not match "official"
 PDQ process and there may be some minuscule difference in score/behaviour.
 
 - `--prob_seg` flag states that detection.json file is formatted for probabilistic segmentation detections as outlined
 above.
 
 - `--mAP_heatmap` flag should not generally be used but enables mAP/moLRP evaluation to be based not upon corners
 defined by PBox/BBox detections, but that encompass all pixels of the detection above given threshold of probability 
 (0.0027). 
 
 For further details, please consult the code.
 
 ### Important Notes ###
 For consistency reasons, unlike the original rvc1 evaluation code, we do not multiply PDQ by 100 to provide it as a percentage.
 PDQ is also labelled as "PDQ" in scores.txt rather than simply "score".
 
 For anyone unfamiliar with moLRP based measures, these values are losses and not qualities like all other provided measures.
 To transform these results from losses to qualities simply take 1 - moLRP.
 
 Newly implemented modes `--segment_mode`, `--bbox_gt`, `greedy_mode` are not used for the RVC1 challenge but can be
 useful for developing research in probabilistic segmentation, when your dataset does not have a segmentation mask, or
 when time is critical, respectively. 
 
visualise_pdq_analysis.py
-------------------------
 To create visualisations for probabilistic detections and PDQ analysis on a single sequence of images run:
 
 `python visualise_pdq_analysis.py --data_type <test_type> --ground_truth <gt_location> --gt_img_folder <gt_imgs_location> --det_json <det_json_file> --gt_analysis <gt_analysis_file> --det_analysis <det_analysis_file> --save_folder <save_folder_location> --set_cov <cov> --img_type <ext> --colour_mode <colour_mode> --corner_mode <corner_mode> --img_set <list_of_img_names> --full_info`
  
 where:
 
 - `<test_type>` is a string defining whether we are evaluating COCO or RVC1 data. Options are 'coco' and 'rvc1'
 
 - `<gt_location>` is a string defining either the location of a ground-truth .json file (coco tests) or a folder of
 ground truth sequences (rvc1 data). Which one it is interpreted as is defined by `<test_type>`
 
 - `<gt_imgs_location>` a string defining the folder where ground-truth images for the sequence are stored.
 
 - `<det_json_file>` a string defining the detection .json file matching the sequence to be visualised
 
 - `<gt_analysis>` a string defining the ground-truth analysis .json file matching the sequence to be visualised.
 Must also correspond to the detection .json file being visualised.
 
 - `<det_analysis>` a string defining the detection analysis .json file matching the sequence to be visualised. 
 Must also correspond to the detection .json file being visualised.
 
 - `<save_folder_location>` a string defining the folder where image visualisations will be saved. Must be different to the `<gt_imgs_location>`
 
 - `<cov>` is an optional value defining set covariance for the corners of the detections. **This must match the set covariance used in evaluate.py**
 
 - `<img_type>` is a string defining what image type the ground-truth is provided in. For example 'jpg'.
 
 - `<colour_mode>` is a string defining whether correct and incorrect results are coloured green and red ('gr') or blue and orange ('bo') respectively.
 Default option is blue and orange.
 
 - `<corner_mode>` is a string defining whether Gaussian corners are represented as three ellipses ('ellipse') or two arrows ('arrow').
 Ellipses are drawn showing 1, 2, and 3, std deviation rings along the contours of the Gaussian. 
 Arrows show 2 x standard deviation along the major axes of the Gaussian.
 Default option is 'ellipse'
 
 - `<list_of_img_names>` is an optional parameter where the user provides a set of image names and only these images will have visualisations drawn for them.
 For example `--img_set cat.jpg dog.jpg whale.jpg` would only draw visualisations for "cat.jpg", "dog.jpg", and "whale.jpg".
 
 - `--full_info` is an optional flag defining whether full pairwise quality analysis should be written for TP detections. **Recommended setting for in-depth analysis**
 
 For further details, please consult the code. 
 
 ### Important Notes ###
 Consistency must be kept between ground-truth analysis, detection analysis, and detection .json files in order to provide meaningful visualisation.
 
 If the evaluation which produced the ground-truth analysis and detection analysis used a set covariance input, you must 
 provide that same set covariance when generating visualisations.
 
 New modes such as using probabilistic segmentation detections (`--prob_seg`) in segment mode (`--segment_mode`)
 or using bounding_box ground-truth (`--bbox_gt`) in the evaluation code are **NOT** yet supported.

visualise_prob_detections.py
-------------------------
 To create visualisations for probabilistic detections on a single sequence of images run:
 
 `python visualise_prob_detections.py --gt_img_folder <gt_imgs_location> --det_json <det_json_file> --save_folder <save_folder_location> --set_cov <cov> --img_type <ext> --corner_mode <corner_mode> --img_set <list_of_img_names>`
  
 where:
 
 - `<gt_imgs_location>` a string defining the folder where ground-truth images for the sequence are stored.
 
 - `<det_json_file>` a string defining the detection .json file matching the sequence to be visualised
 
 - `<save_folder_location>` a string defining the folder where image visualisations will be saved. Must be different to the `<gt_imgs_location>`
 
 - `<cov>` is an optional value defining set covariance for the corners of the detections.
 
 - `<img_type>` is a string defining what image type the ground-truth is provided in. For example 'jpg'.
 
 - `<corner_mode>` is a string defining whether Gaussian corners are represented as three ellipses ('ellipse') or two arrows ('arrow').
 Ellipses are drawn showing 1, 2, and 3, std deviation rings along the contours of the Gaussian. 
 Arrows show 2 x standard deviation along the major axes of the Gaussian.
 Default option is 'ellipse'
 
 - `<list_of_img_names>` is an optional parameter where the user provides a set of image names and only these images will have visualisations drawn for them.
 For example `--img_set cat.jpg dog.jpg whale.jpg` would only draw visualisations for "cat.jpg", "dog.jpg", and "whale.jpg".
 
 For further details, please consult the code. 
 
 ### Important Notes ###
 Order of detections in detections.json file must match the order of the images as stored in the ground-truth images
 folder.
 
 New modes such as using probabilistic segmentation detections (`--prob_seg`) in the evaluation code are 
 **NOT** yet supported.
