#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import os
import time
import logging
import argparse
from collections import defaultdict
from caffe2.python import workspace

from .base import cfgs
from .detectron.core.config import cfg
from .detectron.core.config import assert_and_infer_cfg
from .detectron.core.config import merge_cfg_from_file
from .detectron.utils.logging import setup_logging
from .detectron.core import test_engine as infer_engine
from .detectron.datasets import dummy_datasets as dummy_datasets
from .detectron.utils.timer import Timer
from .detectron.utils.io import cache_url
from .detectron.utils import c2 as c2_utils
from .detectron.utils import vis as vis_utils
from .utils.file_processing import empty_dir


class TableDetector(object):
    def __init__(self):
        c2_utils.import_detectron_ops()
        # OpenCL may be enabled by default in OpenCV3; disable it because it's not
        # thread safe and causes unwanted GPU memory allocations.
        cv2.ocl.setUseOpenCL(False)

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        setup_logging(__name__)
        self.cfgs = cfgs
        self.args = self._parse_args()
        self._config()

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='End-to-end inference')
        parser.add_argument(
            '--cfg',
            dest='cfg',
            help='cfg model file (/path/to/model_config.yaml)',
            default=self.cfgs.YAML_FILE,
            type=str
        )
        parser.add_argument(
            '--wts',
            dest='weights',
            help='weights model file (/path/to/model_weights.pkl)',
            default=self.cfgs.WEIGHTS,
            type=str
        )
        parser.add_argument(
            '--output-dir',
            dest='output_dir',
            help='directory for visualization pdfs (default: /tmp/infer_simple)',
            # default='/tmp/infer_simple',
            default=self.cfgs.OUT_PATH,
            type=str
        )
        parser.add_argument(
            '--image-ext',
            dest='image_ext',
            help='image file name extension (default: jpg)',
            default=self.cfgs.IMAGE_EXT,
            type=str
        )
        parser.add_argument(
            '--always-out',
            dest='out_when_no_box',
            help='output image even when no object is found',
            default=True,
            action='store_true'
        )
        parser.add_argument(
            '--output-ext',
            dest='output_ext',
            help='output image file format (default: pdf)',
            default='pdf',
            type=str
        )
        parser.add_argument(
            '--thresh',
            dest='thresh',
            help='Threshold for visualizing detections',
            default=self.cfgs.DETECT_THRESHOLD,
            type=float
        )
        parser.add_argument(
            '--gpu_id',
            dest='gpu_id',
            help='gpu_id for model',
            default=self.cfgs.GPU_MODE,
            type=int
        )
        parser.add_argument(
            '--kp-thresh',
            dest='kp_thresh',
            help='Threshold for visualizing keypoints',
            default=2.0,
            type=float
        )
        return parser.parse_args()

    def _config(self):
        empty_dir(self.args.output_dir)
        self._logger = logging.getLogger(__name__)

        merge_cfg_from_file(self.args.cfg)
        self.args.weights = cache_url(self.args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        self._detector_model = infer_engine.initialize_model_from_cfg(self.args.weights, gpu_id=self.args.gpu_id)
        self._dummy_coco_dataset = dummy_datasets.get_coco_dataset()
        self._dummy_coco_dataset["classes"][1] = "table"

    def detect_tables(self, im_or_folder):
        if os.path.isdir(im_or_folder):
            im_list = glob.iglob(im_or_folder + '/*.' + self.args.image_ext)
        else:
            im_list = [im_or_folder]

        img_name_list = []
        for i, im_name in enumerate(im_list):
            out_name = os.path.join(
                self.args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + self.args.output_ext)
            )
            if self.cfgs.VERBOSE:
                if self.cfgs.VIS:
                    self._logger.info('Processing {} -> {}'.format(im_name, out_name))
                else:
                    self._logger.info('Processing {}'.format(im_name))
            im = cv2.imread(im_name)
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    self._detector_model, im, None, timers=timers
                )
            if self.cfgs.VERBOSE:
                self._logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                for k, v in timers.items():
                    self._logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            if i == 0:
                self._logger.info(
                    ' \ Note: inference on the first image will be slower than the '
                    'rest (caches and auto-tuning need to warm up)'
                )

            # just for table
            box_list = [b for b in cls_boxes if len(b) > 0]

            target_list = []
            for sub_bbox in box_list:
                for bbox in sub_bbox:
                    if bbox[-1] > self.args.thresh:
                        xmin = int(bbox[1])
                        xmax = int(bbox[3])
                        ymin = int(bbox[0])
                        ymax = int(bbox[2])
                        target_img = im[xmin:xmax, ymin:ymax]
                        target_list.append(target_img)

            for num, save_img in enumerate(target_list):
                file_name, ext = os.path.splitext(os.path.basename(im_name))
                save_name = "{}_{}{}".format(file_name, str(num), ext)
                cv2.imwrite(os.path.join(self.args.output_dir, save_name), save_img)
                img_name_list.append(save_name)

            if self.cfgs.VIS:
                vis_utils.vis_one_image(
                    im[:, :, ::-1],  # BGR -> RGB for visualization
                    im_name,
                    self.args.output_dir,
                    cls_boxes,
                    cls_segms,
                    cls_keyps,
                    dataset=self._dummy_coco_dataset,
                    box_alpha=0.3,
                    show_class=True,
                    thresh=self.args.thresh,
                    kp_thresh=self.args.kp_thresh,
                    ext=self.args.output_ext,
                    out_when_no_box=self.args.out_when_no_box
                )

        # create info.txt
        with open(self.cfgs.IMAGE_INFO, "w") as f:
            for img_name in img_name_list:
                f.write("{}\n".format(img_name))
