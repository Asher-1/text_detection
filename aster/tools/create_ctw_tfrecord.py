#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  aster
FILE_NAME    :  create_ctw_tfrecord
AUTHOR       :  DAHAI LU
TIME         :  2019/8/2 上午11:33
PRODUCT_NAME :  PyCharm
"""

import os
import io
import cv2
import json
import queue
import random
import warnings
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import tensorflow as tf
from PIL import Image, ImageDraw
from aster.utils import dataset_util
from aster.core import standard_fields as fields
from aster.tools import tools

margin_ratio = 0.1
num_keypoints = 128

ROOT_PATH = "/media/yons/data/dataset/images/text_data/ICDAR2017/CTW/data"
OUT_PATH = os.path.join(ROOT_PATH, "ctw.tfrecord")

flags = tf.app.flags
flags.DEFINE_string('data_dir', ROOT_PATH, 'Root directory to raw SynthText dataset.')
flags.DEFINE_integer('start_index', 0, 'Start image index.')
flags.DEFINE_integer('num_images', -1, 'Number of images to create. Default is all remaining.')
flags.DEFINE_integer('shuffle', 0, 'Shuffle images.')
flags.DEFINE_string('output_path', OUT_PATH, 'Path to output TFRecord.')
flags.DEFINE_float('crop_margin', 0, 'margin.')
flags.DEFINE_boolean('write_polygon', False, 'write_polygon.')
flags.DEFINE_integer('num_dump_images', 100, 'Number of images to dump for debugging')
FLAGS = flags.FLAGS


def _fit_and_divide(points):
    # points: [num_points, 2]
    degree = 2 if points.shape[0] > 2 else 1
    coeffs = np.polyfit(points[:, 0], points[:, 1], degree)
    poly_fn = np.poly1d(coeffs)
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    xs = np.linspace(xmin, xmax, num=(num_keypoints // 2))
    ys = poly_fn(xs)
    return np.stack([xs, ys], axis=1)


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # load groundtruth file
    IMAGE_DIR = os.path.join(FLAGS.data_dir, "images/trainval")
    info_annatations = os.path.join(FLAGS.data_dir, "ctw_annotations/info.jsonl")
    train_annatations = os.path.join(FLAGS.data_dir, "ctw_annotations/train.jsonl")
    val_annatations = os.path.join(FLAGS.data_dir, "ctw_annotations/val.jsonl")
    if not os.path.exists(train_annatations):
        raise ValueError('Could not find groundtruth file: {}'.format(train_annatations))
    if not os.path.exists(val_annatations):
        raise ValueError('Could not find groundtruth file: {}'.format(val_annatations))
    print('Loading groundtruth...')

    dummy_text_writer = open(os.path.join("../vis/dummy", "dummy.txt"), "w")
    with open(train_annatations) as f:
        groundtruth = f.read().splitlines()
    with open(val_annatations) as f:
        groundtruth += f.read().splitlines()

    num_images = len(groundtruth) - FLAGS.start_index

    if FLAGS.num_images > 0:
        num_images = min(num_images, FLAGS.num_images)

    indices = list(range(FLAGS.start_index, FLAGS.start_index + num_images))
    if FLAGS.shuffle:
        random.shuffle(indices)

    count = 0
    skipped = 0
    dump_images_count = 0

    for index in tqdm(indices):
        anno = json.loads(groundtruth[index].strip())
        image_id = anno['image_id']
        image_path = os.path.join(IMAGE_DIR, anno['file_name'])

        # load image jpeg data
        im = Image.open(image_path)
        image_w, image_h = im.size
        bbox_array = []
        words = []
        char_polygons_list = []
        for char in tools.each_char(anno):
            if not char['is_chinese']:
                dummy_text_writer.write(char['text'])
                dummy_text_writer.write("\n")
                dummy_text_writer.flush()
                continue
            words.extend(char['text'])
            bbox_x, bbox_y, bbox_w, bbox_h = char['adjusted_bbox']
            if FLAGS.crop_margin > 0:
                margin = bbox_h * FLAGS.crop_margin
                bbox_x = bbox_x - margin
                bbox_y = bbox_y - margin
                bbox_w = bbox_w + 2 * margin
                bbox_h = bbox_h + 2 * margin
                bbox_xmin = int(round(max(0, bbox_x)))
                bbox_ymin = int(round(max(0, bbox_y)))
                bbox_xmax = int(round(min(image_w - 1, bbox_x + bbox_w)))
                bbox_ymax = int(round(min(image_h - 1, bbox_y + bbox_h)))
                bbox_array.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
            else:
                bbox_array.append([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h])

            if FLAGS.write_polygon:
                char_polygons_list.append(char['polygon'])
        bbox_array = np.array(bbox_array)
        if FLAGS.write_polygon:
            char_polygons_list = np.array(char_polygons_list)
        num_bboxes = len(bbox_array)
        if len(words) != num_bboxes:
            raise ValueError('Number of words and bboxes mismtach: {} vs {}'.format(len(words), num_bboxes))

        for i, bbox in enumerate(bbox_array):
            try:
                # crop image and encode to jpeg
                crop_coordinates = tuple(bbox.astype(np.int))
                crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_coordinates
                crop_w, crop_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
                if (crop_xmin < 0 or
                        crop_ymin < 0 or
                        crop_xmax >= image_w or
                        crop_ymax >= image_h or
                        crop_w <= 0 or
                        crop_h <= 0):
                    save_fname = '../vis/invalid/{}_{}.jpg'.format(count, words[i])
                    count += 1
                    bbox_xmin = int(round(max(0, crop_xmin)))
                    bbox_ymin = int(round(max(0, crop_ymin)))
                    bbox_xmax = int(round(min(image_w - 1, crop_xmax)))
                    bbox_ymax = int(round(min(image_h - 1, crop_ymax)))
                    word_crop_im = im.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
                    word_crop_im.save(save_fname)
                    raise ValueError('Invalid crop box {}'.format(crop_coordinates))

                crop_w = crop_xmax - crop_xmin
                crop_h = crop_ymax - crop_ymin
                if crop_w * crop_h < 80:
                    # save_fname = '../vis/invalid/{}_{}.jpg'.format(count, words[i])
                    # count += 1
                    # word_crop_im = im.crop(crop_coordinates)
                    # word_crop_im.save(save_fname)
                    raise ValueError('Crop area too small: {}x{}'.format(crop_w, crop_h))

                word_crop_im = im.crop(crop_coordinates)
                im_buff = io.BytesIO()
                word_crop_im.save(im_buff, format='jpeg')
                word_crop_jpeg = im_buff.getvalue()
                crop_name = '{}:{}'.format(anno['file_name'], i)
                word_crop_w, word_crop_h = word_crop_im.size

                # fit curves to chars polygon points and divide the curve
                flat_curve_points = []
                if FLAGS.write_polygon:
                    char_polygons = char_polygons_list[i]
                    crop_xymin = [crop_xmin, crop_ymin]
                    rel_char_polygons = char_polygons - [[crop_xymin]]
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', np.RankWarning)
                        try:
                            top_curve_points = _fit_and_divide(rel_char_polygons[:, :2, :].reshape([-1, 2]))
                            bottom_curve_points = _fit_and_divide(rel_char_polygons[:, 2:, :].reshape([-1, 2]))
                        except np.RankWarning:
                            raise ValueError('Bad polyfit.')

                    curve_points = np.concatenate([top_curve_points, bottom_curve_points], axis=0)
                    flat_curve_points = curve_points.flatten().tolist()

                if FLAGS.num_dump_images > 0 and dump_images_count < FLAGS.num_dump_images:
                    def _draw_cross(draw, center, size=2):
                        left_pt = tuple(center - [size, 0])
                        right_pt = tuple(center + [size, 0])
                        top_pt = tuple(center - [0, size])
                        bottom_pt = tuple(center + [0, size])
                        draw.line([top_pt, bottom_pt], width=1, fill='#ffffff')
                        draw.line([left_pt, right_pt], width=1, fill='#ffffff')

                    save_fname = '../vis/{}_{}.jpg'.format(count, words[i])
                    draw = ImageDraw.Draw(word_crop_im)
                    if FLAGS.write_polygon:
                        for pts in curve_points:
                            _draw_cross(draw, pts)
                    word_crop_im.save(save_fname)
                    dump_images_count += 1

                example = create_example(crop_name, flat_curve_points, i, word_crop_h, word_crop_jpeg, word_crop_w,
                                         words)

                writer.write(example.SerializeToString())
                count += 1
            except Exception as err:
                print("ValueError: {}".format(err))
                skipped += 1
                continue
                # except:
                #   print("Unexpected error:", sys.exc_info()[0])
                #   skipped += 1
                #   continue

        # except:
        #   print("Image #{} error:".format(index), sys.exc_info()[0])
        #   continue

    print('{} samples created, {} skipped'.format(count, skipped))
    writer.close()
    dummy_text_writer.close()


def create_example(crop_name, flat_curve_points, i, word_crop_h, word_crop_jpeg, word_crop_w, words):
    # write an example
    if FLAGS.write_polygon:
        example = tf.train.Example(features=tf.train.Features(feature={
            fields.TfExampleFields.image_encoded: \
                dataset_util.bytes_feature(word_crop_jpeg),
            fields.TfExampleFields.image_format: \
                dataset_util.bytes_feature('jpeg'.encode('utf-8')),
            fields.TfExampleFields.filename: \
                dataset_util.bytes_feature(crop_name.encode('utf-8')),
            fields.TfExampleFields.channels: \
                dataset_util.int64_feature(3),
            fields.TfExampleFields.colorspace: \
                dataset_util.bytes_feature('rgb'.encode('utf-8')),
            fields.TfExampleFields.width: \
                dataset_util.int64_feature(word_crop_w),
            fields.TfExampleFields.height: \
                dataset_util.int64_feature(word_crop_h),
            fields.TfExampleFields.transcript: \
                dataset_util.bytes_feature(words[i].encode('utf-8')),
            fields.TfExampleFields.keypoints: \
                dataset_util.float_list_feature(flat_curve_points),
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            fields.TfExampleFields.image_encoded: \
                dataset_util.bytes_feature(word_crop_jpeg),
            fields.TfExampleFields.image_format: \
                dataset_util.bytes_feature('jpeg'.encode('utf-8')),
            fields.TfExampleFields.filename: \
                dataset_util.bytes_feature(crop_name.encode('utf-8')),
            fields.TfExampleFields.channels: \
                dataset_util.int64_feature(3),
            fields.TfExampleFields.colorspace: \
                dataset_util.bytes_feature('rgb'.encode('utf-8')),
            fields.TfExampleFields.width: \
                dataset_util.int64_feature(word_crop_w),
            fields.TfExampleFields.height: \
                dataset_util.int64_feature(word_crop_h),
            fields.TfExampleFields.transcript: \
                dataset_util.bytes_feature(words[i].encode('utf-8')),
        }))
    return example


if __name__ == '__main__':
    tf.app.run()
