Overview
========
This code is hacky messy code that enables the evaluation of detectors on Common Objects in Context (*COCO*) and 
Robotic Vision Challenge 1 (*RVC1*) data. Evaluation is performed using mean average precision (*mAP*) as implemented in
the COCO challenge and probabilistic detection quality (*PDQ*) as implemented (mostly) in the RVC1 CVPR2019 challenge.

Setup
=====
After installing all requirements, you will need to have a fully installed implementation of the COCO API located
somewhere on your machine.
You can download this API here https://github.com/cocodataset/cocoapi.

Once this is downloaded and installed, you need to adjust the system path on line 7 of coco_mAP.py and line 9 of 
read_files.py to match the PythonAPI folder of your COCO API installation.

Usage
=====
All evaluation code is run on detections saved in .json files formatted as required by the RVC.
If you are evaluating on COCO data and have saved detections in COCO format, you can convert to RVC format using 
*convert_coco.py*
When you have the appropriate files, you can evaluate on mAP and PDQ with *evaluate.py*.
After evaluation is complete, you can visualise your detections for a sequence of images w.r.t. PDQ using 
*visualise_pdq_analysis.py*

convert_coco.py
---------------
 To convert coco detections to rvc format simply run:
 
 `python convert_coco.py --coco_gt <gt_json_file> --coco_det <det_json_file> --rvc1_det <output_json_file>`
 
 where `<gt_json_file>` is the coco format ground-truth json filename, `det_json_file` is the coco format detection 
 json filename, and `output_file` is the json filename you will save your rvc1 formatted detections json file.
 
 evaluate.py
 -----------
 To perform full evaluation simply run:
 
 `python evaluate.py --test_set <test_type> --gt_loc <gt_location> --det_loc <det_location> --save_folder <save_folder> --set_cov <cov>`
 
 There is also an `--mAP_heatmap` flag but that should not generally be used.
 
 `<test_type>` is a string defining whether we are evaluating COCO or rvc1 data. Options are 'coco' and 'rvc1'
 
 `<gt_location>` is a string defining either the location of a ground-truth .json file (coco tests) or a folder of
 ground truth sequences (rvc1 data). Which one it is interpreted as is defined by `<test_type>`
 
 `<det_loc>` is a string defining either the location of a detection .json file (coco data) or a folder of .json files for 
 multiple sequences (rvc1 data). Which one it is interpreted as is defined by `<test_type>`.
 Note that these detection files must be in rvc1 format.
 
 `<save_folder>` is a string defining the folder where analysis will be stored in form of scores.txt, and files for visualisations
 `<cov>` is an optional value defining set covariance for the corners of detections. If not using a detector with  calculated
 covariances this needs to be manually set to 0.
 
 visualise_pdq_analysis.py
 -------------------------
 To create visualisations for PDQ detections simply run:
 
 `python visualise_pdq_analysis.py --data_type <test_type> --ground_truth <gt_location> --gt_img_folder <gt_imgs_location> --det_json <det_json_file> --det_analysis <det_analysis_file> --gt_analysis <gt_analysis_file> --save_folder <save_folder_location> --set_cov <cov> --img_type <ext> --full_info (opt)` 
 Recommend running with full_info flag on.
 TODO add description ... too sick ... read code for parameter meanings if you need it that badly.
