import unittest
import numpy as np
from data_holders import BBoxDetInst, GroundTruthInstance, PBoxDetInst
from pdq import PDQ
_SMALL_VAL = 1e-14
_MAX_LOSS = np.log(_SMALL_VAL)


class TestPDQ(unittest.TestCase):

    def setUp(self):
        self.img_size = (2000, 2000)
        self.default_covar = [[1000, 0], [0, 1000]]
        self.default_filter_gt = False
        self.default_segment_mode = False
        self.default_greedy_mode = False

        self.square_mask = np.zeros(self.img_size, dtype=np.bool)
        self.square_mask[750:1250, 750:1250] = True
        self.square_gt = [GroundTruthInstance(self.square_mask, 0)]
        self.square_gt_box = self.square_gt[0].bounding_box
        self.square_label_list = [1, 0]

        self.cross_mask = np.zeros(self.img_size, dtype=np.bool)
        self.cross_mask[875:1125, 750:1250] = True
        self.cross_mask[750:1250, 875:1125] = True
        self.cross_gt = [GroundTruthInstance(self.cross_mask, 1)]
        self.cross_gt_box = self.cross_gt[0].bounding_box
        self.cross_label_list = [0, 1]

    def test_perfect_bbox(self):
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(score, 1, 4)

    def test_no_detection(self):
        detections = []
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertEqual(score, 0)

    def test_detection_no_gt(self):
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        gts = []
        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertEqual(score, 0)

    def test_half_label_confidence(self):
        detections = [BBoxDetInst([0.5, 0.5], self.square_gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(np.sqrt(0.5), score, 4)

    def test_half_position_confidence(self):
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box, 0.5)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(np.sqrt(0.5), score, 4)

    def test_half_position_and_label_confidences(self):
        detections = [BBoxDetInst([0.5, 0.5], self.square_gt_box, 0.5)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        self.assertAlmostEqual(0.5, score, 4)

    def test_miss_500_pixels(self):
        det_box = [val for val in self.square_gt_box]
        det_box[2] -= 1
        detections = [BBoxDetInst(self.square_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*500)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_detect_500_extra_pixels(self):
        det_box = [val for val in self.square_gt_box]
        det_box[2] += 1
        detections = [BBoxDetInst(self.square_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*500)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_shift_right_one_pixel(self):
        det_box = [val for val in self.square_gt_box]
        det_box[2] += 1
        det_box[0] += 1
        detections = [BBoxDetInst(self.square_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = np.exp((_MAX_LOSS*1000)/(500*500))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_run_score_multiple_times_on_single_pdq_instance(self):
        two_detections = [BBoxDetInst(self.square_label_list, self.square_gt_box),
                          BBoxDetInst(self.square_label_list, self.square_gt_box)]
        one_detection = [BBoxDetInst(self.square_label_list, self.square_gt_box)]

        evaluator = PDQ()
        score_two = evaluator.score([(self.square_gt, two_detections)])
        score_one = evaluator.score([(self.square_gt, one_detection)])

        self.assertAlmostEqual(score_two, 0.5)
        self.assertAlmostEqual(score_one, 1.0)

    def test_no_overlap_box(self):
        det_box = [val for val in self.square_gt_box]
        box_width = (self.square_gt_box[2] + 1) - self.square_gt_box[0]
        det_box[0] += box_width
        det_box[2] += box_width

        detections = [BBoxDetInst(self.square_label_list, det_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, detections)])

        expected_spatial_quality = 0

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(expected_gmean, score, 4)

    def test_multiple_detections(self):
        ten_detections = [BBoxDetInst(self.square_label_list, self.square_gt_box) for _ in range(10)]
        evaluator = PDQ()
        score = evaluator.score([(self.square_gt, ten_detections)])

        self.assertAlmostEqual(score, 0.1)

    def test_multiple_missed_gts(self):

        gts = [val for val in self.square_gt]
        for i in range(9):
            # Create small 11x11 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:14, 2 + i * 14:14 + i * 14] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0))
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 0.1)

    def test_multiple_too_small_gts_filtered(self):

        gts = [val for val in self.square_gt]
        for i in range(9):
            # Create small 2x2 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:4, 2 + i * 4:4 + i * 4] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0))
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        evaluator = PDQ(filter_gts=True)
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 1.0)

    def test_multiple_too_small_gts_not_filtered(self):

        gts = [val for val in self.square_gt]
        for i in range(9):
            # Create small 2x2 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:4, 2 + i * 4:4 + i * 4] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0))
        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        evaluator = PDQ(filter_gts=False)
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 0.1)

    def test_missed_gts_and_unmatched_detections(self):
        gts = [val for val in self.square_gt]
        for i in range(10):
            # Create small 11x11 boxes which are missed around an edge of the image (buffer of 2 pixels)
            new_gt_mask = np.zeros(gts[0].segmentation_mask.shape, gts[0].segmentation_mask.dtype)
            new_gt_mask[2:14, 2 + i * 14:14 + i * 14] = np.amax(gts[0].segmentation_mask)
            gts.append(GroundTruthInstance(new_gt_mask, 0))

        detections = [BBoxDetInst(self.square_label_list, self.square_gt_box) for _ in range(10)]

        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 1/20.)

    def test_correct_second_detection(self):
        gts = [val for val in self.square_gt]
        detections = [BBoxDetInst(self.square_label_list, [0, 0, 10, 10]),
                      BBoxDetInst(self.square_label_list, self.square_gt_box)]

        evaluator = PDQ()
        score = evaluator.score([(gts, detections)])

        self.assertAlmostEqual(score, 0.5)

    def test_no_detections_for_image(self):
        gts1 = [val for val in self.square_gt]
        gts2 = [GroundTruthInstance(self.square_mask, 0)]
        dets1 = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        dets2 = []
        evaluator = PDQ()
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 0.5)

    def test_no_detections_for_image_with_filtered_small_gt(self):
        gts1 = [val for val in self.square_gt]
        small_mask = np.zeros(self.img_size, dtype=np.bool)
        small_mask[500:504, 500:501] = True
        gts2 = [GroundTruthInstance(small_mask, 0)]
        dets1 = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        dets2 = []
        evaluator = PDQ(filter_gts=True)
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 1.0)

    def test_no_detections_for_image_with_small_and_big_gt_with_filtering(self):
        gts1 = [val for val in self.square_gt]
        small_mask = np.zeros(self.img_size, dtype=np.bool)
        small_mask[500:504, 500:501] = True
        gts2 = [GroundTruthInstance(self.square_mask, 0),
                GroundTruthInstance(small_mask, 0)]
        dets1 = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
        dets2 = []
        evaluator = PDQ(filter_gts=True)
        score = evaluator.score([(gts1, dets1), (gts2, dets2)])

        self.assertAlmostEqual(score, 0.5)

    # Removed this functionality. Not deemed necessary
    # def test_filter_small_for_one_image_and_not_other(self):
    #     small_mask = np.zeros(self.img_size, dtype=np.bool)
    #     small_mask[500:504, 500:501] = True
    #     gts1 = [GroundTruthInstance(self.square_mask, 0),
    #             GroundTruthInstance(small_mask, 0)]
    #     gts2 = [GroundTruthInstance(self.square_mask, 0),
    #             GroundTruthInstance(small_mask, 0)]
    #     dets1 = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
    #     dets2 = [BBoxDetInst(self.square_label_list, self.square_gt_box)]
    #     evaluator = PDQ()
    #     score = evaluator.score([(gts1, dets1, True, self.default_segment_mode, self.default_greedy_mode),
    #                              (gts2, dets2, False, self.default_segment_mode, self.default_greedy_mode)])
    #
    #     self.assertAlmostEqual(score, 2./3)

    def test_cross_gt_detected_by_perfect_box_in_non_segment_mode(self):
        detections = [BBoxDetInst(self.cross_label_list, self.cross_gt_box)]
        evaluator = PDQ()
        score = evaluator.score([(self.cross_gt, detections)])

        self.assertAlmostEqual(score, 1, 4)

    def test_cross_gt_detected_by_perfect_box_in_segment_mode(self):
        detections = [BBoxDetInst(self.cross_label_list, self.cross_gt_box)]
        evaluator = PDQ(segment_mode=True)
        score = evaluator.score([(self.cross_gt, detections)])

        expected_num_missed_pixels = np.sum(np.logical_xor(self.square_mask, self.cross_mask))
        expected_spatial_quality = np.exp((_MAX_LOSS * expected_num_missed_pixels) / np.sum(self.cross_mask))

        expected_gmean = np.sqrt(expected_spatial_quality)

        self.assertAlmostEqual(score, expected_gmean, 4)

    def test_hungarian_assignment(self):
        gt_big_mask = self.square_mask.copy()
        gt_small_mask = np.zeros(gt_big_mask.shape, dtype=gt_big_mask.dtype)
        gt_small_mask[900:1101, 900:1101] = True

        gt_big = GroundTruthInstance(gt_big_mask, 0)
        gt_small = GroundTruthInstance(gt_small_mask, 0)
        gts = [gt_big, gt_small]
        # Big Det
        det1 = PBoxDetInst(self.square_label_list, [700, 700, 1300, 1300],
                           [[[10000, 0], [0, 10000]], [[10000, 0], [0, 10000]]])
        # Small Det
        det2 = PBoxDetInst(self.square_label_list, [800, 800, 1200, 1200],
                           [[[10000, 0], [0, 10000]], [[10000, 0], [0, 10000]]])

        dets = [det1, det2]

        evaluator = PDQ(greedy_mode=False)
        evaluator.score([(gts, dets)])
        det_evals = evaluator._det_evals
        for img_det_evals in det_evals:
            for det_eval in img_det_evals:
                # Assume that big detection should be matched to big gt and small to small
                self.assertEqual(det_eval['det_id'], det_eval['gt_id'])

    def test_greedy_assignment(self):
        gt_big_mask = self.square_mask.copy()
        gt_small_mask = np.zeros(gt_big_mask.shape, dtype=gt_big_mask.dtype)
        gt_small_mask[900:1101, 900:1101] = True

        gt_big = GroundTruthInstance(gt_big_mask, 0)
        gt_small = GroundTruthInstance(gt_small_mask, 0)
        gts = [gt_big, gt_small]
        # Big Det
        det1 = PBoxDetInst(self.square_label_list, [700, 700, 1300, 1300],
                           [[[10000, 0], [0, 10000]], [[10000, 0], [0, 10000]]])
        # Small Det
        det2 = PBoxDetInst(self.square_label_list, [800, 800, 1200, 1200],
                           [[[10000, 0], [0, 10000]], [[10000, 0], [0, 10000]]])

        dets = [det1, det2]

        evaluator = PDQ(greedy_mode=True)
        evaluator.score([(gts, dets)])
        det_evals = evaluator._det_evals
        for img_det_evals in det_evals:
            for det_eval in img_det_evals:
                # Assume that big detection should be matched to small gt and small det to big gt
                self.assertNotEqual(det_eval['det_id'], det_eval['gt_id'])


