import numpy as np


class CECalculator(object):
    def __init__(self, num_bins=10, ce_max_class=False, ce_no_fn=False):
        self.num_bins = num_bins
        self.ce_max_class = ce_max_class
        self._ce_no_fn=ce_no_fn
        # decree bins like this to have num_bins equally spaced between zero and one (we will ignore bin zero)
        self.bins = np.linspace(0, 1, self.num_bins+1)

        self._avg_confs = np.zeros(self.num_bins)
        self._num_dets_per_bin = np.zeros(self.num_bins)
        self._num_correct_per_bin = np.zeros(self.num_bins)

    def add_img(self, gt_labels, det_probs):
        """
        Update calibration error information for a new image's worth of observations
        :param gt_labels: d list of gt_labels for all detections (missed detections have gt c+1)
        :param det_probs: dxc+1 numpy array of probability distributions across all classes (false negatives are 1.0 class c+1)
        Note that the det_probs and gt_labels must correspond precisely
        :return: None
        """
        # TODO Update ALL DOCUMENTATION to avoid confusion when reviewing later on
        # TODO ALL of this can no doubt be optimized later on
        # Allocate detections and gts to bin based on probability
        # Note we count all probabilities in distribution as a single detection

        # bin 0 contains only detections with probability 0 (i.e. they don't exist). This bin is ignored in analysis
        # Intervals include right bin edge
        bin_ids = np.digitize(det_probs, self.bins, True)  # d x c+1
        max_classes = np.argmax(det_probs, axis=1)  # d

        if not self.ce_max_class:
            # note that bin_ids should be ints between zero and self.num_bins-1
            for bin_id in np.unique(bin_ids):
                # skip analysing detections with probability zero. These do not count as detections
                if bin_id == 0:
                    continue


                current_num_bin_dets = np.sum(bin_ids == bin_id)
                self._num_dets_per_bin[bin_id - 1] += current_num_bin_dets

                # Calculate the average confidence for detections in this bin (using incremental averaging)
                img_avg_conf = np.mean(det_probs[bin_ids == bin_id])
                current_det_ratio = current_num_bin_dets / self._num_dets_per_bin[bin_id - 1]
                self._avg_confs[bin_id - 1] += current_det_ratio * (img_avg_conf - self._avg_confs[bin_id - 1])

        # Calculate the number of ground-truths of each class allocated to each bin
        # Note that a new "ground-truth" is created for each detection's probability scores
        # TODO Optimize with numpy if possible later
        for det_idx, det_bin_ids in enumerate(bin_ids):

            correct_gt = gt_labels[det_idx]

            if self.ce_max_class:
                # Calculate the average confidences for detections in the bin of the given detection
                bin_id = det_bin_ids[max_classes[det_idx]]
                if bin_id != 0:
                    self._num_dets_per_bin[bin_id - 1] += 1
                    self._avg_confs[bin_id - 1] += det_probs[det_idx, max_classes[det_idx]] - self._avg_confs[bin_id - 1]
                    if max_classes[det_idx] == correct_gt:
                        self._num_correct_per_bin[bin_id - 1] += 1

            else:
                correct_gt_bin = det_bin_ids[correct_gt]
                if correct_gt_bin != 0:
                    self._num_correct_per_bin[correct_gt_bin - 1] += 1

    def add_pPDQ_img(self, assignments, ppdq_mat, img_gt_labels, img_det_probs, img_gt_ignores):
        # Note num dets total here is number of dets including "background" detections for FNs
        calib_gt_labels = []    # will be num dets total long
        calib_det_probs = None  # will be num dets total x c+1

        # Calulate the inputs to be used for add_img
        # need to match gt_labels as assigned to det_probs
        # false positives must be assigned to gt_label c
        # false negatives must be included as all zero det_probs
        for gt_idx, det_idx in assignments:
            # Skip pairing if gt index should be ignored
            if gt_idx in img_gt_ignores:
                continue
            # TODO compact this with better logic rather than TP, FP, FN split

            # True positive assignments
            if ppdq_mat[gt_idx, det_idx] > 0:
                if calib_det_probs is None:
                    calib_det_probs = np.append(img_det_probs[det_idx], 0)[np.newaxis]    # 1xc+1
                else:
                    calib_det_probs = np.vstack((calib_det_probs, np.append(img_det_probs[det_idx], 0)[np.newaxis]))
                calib_gt_labels.append(img_gt_labels[gt_idx])

            else:
                # False positive assignments
                if det_idx < img_det_probs.shape[0]:
                    if calib_det_probs is None:
                        calib_det_probs = np.append(img_det_probs[det_idx], 0)[np.newaxis]  # 1xc+1
                    else:
                        calib_det_probs = np.vstack((calib_det_probs, np.append(img_det_probs[det_idx], 0)[np.newaxis]))

                    # false positives are gt_label c
                    calib_gt_labels.append(img_det_probs.shape[1])

                # False negative assignments
                if gt_idx < len(img_gt_labels):
                    # Ignore FN assignments if told to
                    if not self._ce_no_fn:
                        # Deal with image with no detections
                        if img_det_probs.shape[0] == 0:
                            det_prob_shape = np.amax(img_gt_labels)
                        else:
                            det_prob_shape = img_det_probs.shape[1]

                        # False negatives are "detections" with 100% confidence background
                        if calib_det_probs is None:
                            calib_det_probs = np.zeros((1, det_prob_shape + 1))     # 1 x c+1
                        else:
                            calib_det_probs = np.vstack((calib_det_probs, np.zeros((1, det_prob_shape + 1))))
                        calib_det_probs[-1, -1] = 1.0

                        calib_gt_labels.append(img_gt_labels[gt_idx])

        # Add image info to the algorithm as normal if information has not been fully ignored
        if len(calib_gt_labels) != 0:
            self.add_img(calib_gt_labels, calib_det_probs)

    def get_expected_calibration_error(self):
        ece_score = 0
        for bin_id in range(self.num_bins):
            # if there are no detections for this bin continue
            # in this case average should be zero too
            if self._num_dets_per_bin[bin_id] == 0:
                continue
            bin_accuracy = self._num_correct_per_bin[bin_id] / float(self._num_dets_per_bin[bin_id])
            bin_ratio = self._num_dets_per_bin[bin_id] / float(np.sum(self._num_dets_per_bin))
            ece_score += bin_ratio * abs(bin_accuracy - self._avg_confs[bin_id])

        return ece_score

    def get_maximum_calibration_error(self):
        mce_score = 0
        for bin_id in range(self.num_bins):
            bin_accuracy = self._num_correct_per_bin[bin_id] / float(self._num_dets_per_bin[bin_id])

            mce_score = max(mce_score, abs(bin_accuracy - self._avg_confs[bin_id]))

        return mce_score