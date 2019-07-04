import unittest
import numpy as np
from data_holders import BBoxDetInst, PBoxDetInst, GroundTruthInstance
import os
import shutil
import json
import read_files

class TestReadPBoxJson(unittest.TestCase):

    def setUp(self):
        self.det_files_root = '/tmp/pdq_utest_read_files/'
        if not os.path.isdir(self.det_files_root):
            os.makedirs(self.det_files_root)

        self.default_det_dict = {'bbox': [750, 750, 1250, 1250], "label_probs": [0, 1, 0]}
        self.default_covar = [[1000, 0], [0, 1000]]
        self.defaul_det_classes = ['background', 'square', 'cross']
        self.default_img_names = ['square.jpg']
        self.default_gt_classes = ['none', 'square', 'cross', 'diamond']
        self.default_gt_class_ids_dict = {'none': 0, 'square': 1, 'cross': 2, 'diamond': 3}

    def tearDown(self):
        if os.path.isdir(self.det_files_root):
            shutil.rmtree(self.det_files_root)

    def test_single_pbox(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_two_pboxes_one_img(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict, self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 2)
        self.assertIsInstance(img_dets[0], PBoxDetInst)
        self.assertIsInstance(img_dets[1], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])
        self.assertTrue(np.allclose(img_dets[1].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[1].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[1].class_list), self.default_det_dict['label_probs'])

    def test_two_pboxes_two_imgs(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict], [self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 2)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_empty_img(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict], []]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 2)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 0)

    def test_single_pbox_w_extra_gt_label(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name, self.default_gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = [0, 1, 0, 0]
        self.assertListEqual(list(img_dets[0].class_list), expected_label_probs)

    def test_single_pbox_w_rearranged_gt_label(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        gt_class_ids_dict = {'none': 0, 'cross': 1, 'square': 2}
        detections = read_files.read_pbox_json(det_file_name, gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = [0, 0, 1]
        self.assertListEqual(list(img_dets[0].class_list), expected_label_probs)

    def test_single_pbox_w_fewer_gt_labels(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        gt_class_ids_dict = {'square': 0, 'cross': 1}
        detections = read_files.read_pbox_json(det_file_name, gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = [1, 0]
        self.assertListEqual(list(img_dets[0].class_list), expected_label_probs)

    def test_single_pbox_w_gt_label_id_skipped(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        gt_class_ids_dict = {'square': 1, 'cross': 4}
        detections = read_files.read_pbox_json(det_file_name, gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = [0, 1, 0, 0, 0]
        self.assertListEqual(list(img_dets[0].class_list), expected_label_probs)

    def test_single_pbox_w_synonym_label_id(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        self.default_det_dict.update({'label_probs': [0.3, 0.7, 0]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        gt_class_ids_dict = {'none': 0, 'cross': 1, 'square': 2}
        detections = read_files.read_pbox_json(det_file_name, gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = np.array([0.3, 0.0, 0.7])
        self.assertTrue(np.allclose(img_dets[0].class_list, expected_label_probs))

    def test_single_pbox_w_non_gt_class(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        self.default_det_dict.update({'label_probs': [0, 0.7, 0.3]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        gt_class_ids_dict = {'none': 0, 'diamond': 1, 'square': 2}
        detections = read_files.read_pbox_json(det_file_name, gt_class_ids_dict)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        expected_label_probs = np.array([0.0, 0.0, 0.7])
        self.assertTrue(np.allclose(img_dets[0].class_list, expected_label_probs))

    def test_single_bbox(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_single_bbox_0_covar_in_file(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_single_bbox_set_covar_0(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_single_bbox_set_covar_0_base_covar_non_zero(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name, override_cov=0)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_single_pbox_set_covar(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names
        det_data['detections'] = [[self.default_det_dict]]
        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name, override_cov=200)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [[[200, 0], [0, 200]], [[200, 0], [0, 200]]]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

    def test_one_pbox_one_bbox_one_img(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names

        # Create one bbox detection and one pbox detection
        box_det_dict = self.default_det_dict.copy()
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict, box_det_dict]]

        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections, read_files.BoxLoader)

        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 2)
        self.assertIsInstance(img_dets[0], PBoxDetInst)
        self.assertIsInstance(img_dets[1], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertTrue(np.allclose(img_dets[1].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])
        self.assertListEqual(list(img_dets[1].class_list), self.default_det_dict['label_probs'])

    def test_one_pbox_one_bbox_diff_imgs(self):
        # Create detection.json file
        det_data = {}
        det_data['classes'] = self.defaul_det_classes
        det_data['img_names'] = self.default_img_names

        # Create one bbox detection and one pbox detection
        box_det_dict = self.default_det_dict.copy()
        self.default_det_dict.update({'covars': [self.default_covar, self.default_covar]})
        det_data['detections'] = [[self.default_det_dict], [box_det_dict]]

        det_file_name = os.path.join(self.det_files_root, 'det_file.json')
        with open(det_file_name, 'w') as f:
            json.dump(det_data, f)

        detections = read_files.read_pbox_json(det_file_name)

        self.assertEqual(len(detections), 2)
        self.assertIsInstance(detections, read_files.BoxLoader)

        # Image 1
        img_iterator = iter(detections)
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], PBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].covs, [self.default_covar, self.default_covar]))
        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])

        # Image 2
        img_dets = next(img_iterator)

        self.assertEqual(len(img_dets), 1)
        self.assertIsInstance(img_dets[0], BBoxDetInst)

        self.assertTrue(np.allclose(img_dets[0].box, self.default_det_dict['bbox']))
        self.assertListEqual(list(img_dets[0].class_list), self.default_det_dict['label_probs'])




