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

ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data"
DATA_DIR = os.path.join(ROOT_PATH, "images")
VIS_DIR = os.path.join(ROOT_PATH, "vis")
# ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_test.txt")
ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_train.txt")
map_path = os.path.join(ROOT_PATH, "char_std_5990.txt")

OUT_DIR = os.path.join(ROOT_PATH, "tfrecord/syn_300k.tfrecord")
VIS = os.path.join(ROOT_PATH, "vis")

map_path = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/char_std_5990.txt"

flags = tf.app.flags
flags.DEFINE_string('data_dir', DATA_DIR, 'Root directory to raw SynthText dataset.')
flags.DEFINE_string('out_dir', OUT_DIR, 'tfrecord file path.')
flags.DEFINE_string('vis_dir', "", 'VIS path.')
flags.DEFINE_float('crop_margin', 0.05, 'Margin in percentage of word height')
FLAGS = flags.FLAGS


def index_to_string(charset, texts):
    labels = ""
    for text in texts:
        labels += charset[int(text)]
    return labels


def readfile(filename):
    with open(map_path, 'r', encoding='utf-8') as f:
        char_set = f.read().splitlines()
    with open(filename, 'r') as f:
        res = f.read().splitlines()
    dic = {}
    for i in res:
        p = i.split(' ')
        char_label = index_to_string(char_set, p[1:])
        dic[p[0]] = char_label
    return dic


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.out_dir)
    count = 0
    data_dir = FLAGS.data_dir
    file_list = readfile(ANNOTATION_PATH)
    for file_name, text in tqdm(list(file_list.items())):
        image_id = os.path.splitext(file_name)[0]
        image_path = os.path.join(data_dir, file_name)
        image = Image.open(image_path)
        try:
            if sys.version_info.major == 2:
                text = text.replace('\xef\xbb\bf', '')
                text = text.replace('\xe2\x80\x8d', '')
            else:
                text = text.replace('\ufeff', '')
                text = text.replace('\xef\xbb\xbf', '')
            groundtruth_text = text.strip()
            if FLAGS.vis_dir != "":
                save_fname = '{}/{}.jpg'.format(FLAGS.vis_dir, groundtruth_text.replace(r"/", ""))
                image.save(save_fname)

            im_buff = io.BytesIO()
            image.save(im_buff, format='jpeg')
            word_crop_jpeg = im_buff.getvalue()
            crop_name = '{}:{}'.format(file_name, count)

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
