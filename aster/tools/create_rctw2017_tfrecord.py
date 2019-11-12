import os
import io
import re
import sys
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

ROOT_PATH = "/media/yons/data/dataset/images/text_data/ICDAR2017/RCTW/"
DATA_DIR = ROOT_PATH + "train"
OUT_DIR = ROOT_PATH + "rctw2.tfrecord"
VIS = ROOT_PATH + "vis"

flags = tf.app.flags
flags.DEFINE_string('data_dir', DATA_DIR, 'Root directory to raw SynthText dataset.')
flags.DEFINE_string('out_dir', OUT_DIR, 'tfrecord file path.')
flags.DEFINE_string('vis_dir', "", 'VIS path.')
flags.DEFINE_float('crop_margin', 0.05, 'Margin in percentage of word height')
FLAGS = flags.FLAGS


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.out_dir)
    count = 0
    groundtruth_dir = FLAGS.data_dir
    groundtruth_files = glob.glob(os.path.join(groundtruth_dir, '*.txt'))

    for groundtruth_file in tqdm(groundtruth_files):
        image_id = os.path.basename(groundtruth_file).split(".")[0]
        image_rel_path = '{}.jpg'.format(image_id)
        image_path = os.path.join(groundtruth_dir, image_rel_path)
        image = Image.open(image_path)
        image_w, image_h = image.size

        with open(groundtruth_file, 'r') as f:
            groundtruth = f.read().splitlines()

        for i, line in enumerate(groundtruth):
            try:
                if sys.version_info.major == 2:
                    line = line.replace('\xef\xbb\bf', '')
                    line = line.replace('\xe2\x80\x8d', '')
                else:
                    line = line.replace('\ufeff', '')
                    line = line.replace('\xef\xbb\xbf', '')
                info_list = line.split(",")
                coord = list(map(float, info_list[:8]))
                coord_array = np.array(coord).reshape((-1, 2))
                bbox_xmin = np.min(coord_array[:, 0])
                bbox_ymin = np.min(coord_array[:, 1])
                bbox_xmax = np.max(coord_array[:, 0])
                bbox_ymax = np.max(coord_array[:, 1])
                difficulty = info_list[8]
                groundtruth_text = info_list[9].split("\"")[1]

                if difficulty == "1":
                    continue

                if FLAGS.crop_margin > 0:
                    bbox_h = bbox_ymax - bbox_ymin
                    margin = bbox_h * FLAGS.crop_margin
                    bbox_xmin = bbox_xmin - margin
                    bbox_ymin = bbox_ymin - margin
                    bbox_xmax = bbox_xmax + margin
                    bbox_ymax = bbox_ymax + margin
                bbox_xmin = int(round(max(0, bbox_xmin)))
                bbox_ymin = int(round(max(0, bbox_ymin)))
                bbox_xmax = int(round(min(image_w - 1, bbox_xmax)))
                bbox_ymax = int(round(min(image_h - 1, bbox_ymax)))

                crop_w, crop_h = bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin
                if crop_w * crop_h < 80:
                    print("the area is too small : {}".format(crop_w * crop_h))
                    continue
                try:
                    word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
                except Exception as e:
                    print("invalid box: ", (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
                    print(e)
                    continue
                if FLAGS.vis_dir != "":
                    save_fname = '{}/{}_{}.jpg'.format(FLAGS.vis_dir, groundtruth_text.replace(r"/", ""), i)
                    word_crop_im.save(save_fname)

                im_buff = io.BytesIO()
                word_crop_im.save(im_buff, format='jpeg')
                word_crop_jpeg = im_buff.getvalue()
                crop_name = '{}:{}'.format(image_rel_path, i)

                label = "^".join(groundtruth_text)
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
                    fields.TfExampleFields.transcript: \
                        dataset_util.bytes_feature(label.encode('utf-8')),
                }))
                writer.write(example.SerializeToString())
                count += 1
            except Exception as e:
                print(e)
                continue
    writer.close()
    print('{} examples created'.format(count))


if __name__ == '__main__':
    tf.app.run()
