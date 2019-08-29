# -*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine.topology import preprocess_weights_for_loading
import h5py
from imp import reload
import densenet


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if len(str) <= 0:
                print("len <= 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


def load_models(file_path, model):
    def load_weights_from_hdf5_group_by_name(f, layers):
        """Implements name-based weight loading.

        (instead of topological weight loading).

        Layers that have no matching name are skipped.

        # Arguments
            f: A pointer to a HDF5 group.
            layers: a list of target layers.

        # Raises
            ValueError: in case of mismatch between provided layers
                and weights file.
        """
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None

        # New file format.
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # Reverse index of layer name to list of layers with name.
        index = {}
        for layer in layers:
            if layer.name:
                index.setdefault(layer.name, []).append(layer)

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            if weight_names == ['out_2/kernel:0', 'out_2/bias:0']:
                print("skip {}".format(weight_names))
                continue

            for layer in index.get(name, []):
                symbolic_weights = layer.weights
                weight_values = preprocess_weights_for_loading(
                    layer,
                    weight_values,
                    original_keras_version,
                    original_backend)
                if len(weight_values) != len(symbolic_weights):
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
                # Set values.
                for i in range(len(weight_values)):
                    weight_value_tuples.append((symbolic_weights[i],
                                                weight_values[i]))
        K.batch_set_value(weight_value_tuples)

    def load_weights_from_hdf5_group(f, layers):
        """Implements topological (order-based) weight loading.

        # Arguments
            f: A pointer to a HDF5 group.
            layers: a list of target layers.

        # Raises
            ValueError: in case of mismatch between provided layers
                and weights file.
        """
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None

        filtered_layers = []
        for layer in layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            print(name)
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if len(layer_names) != len(filtered_layers):
            raise ValueError('You are trying to load a weight file '
                             'containing ' + str(len(layer_names)) +
                             ' layers into a model with ' +
                             str(len(filtered_layers)) + ' layers.')

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = filtered_layers[k]
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(layer,
                                                           weight_values,
                                                           original_keras_version,
                                                           original_backend)
            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '" in the current model) was found to '
                                 'correspond to layer ' + name +
                                 ' in the save file. '
                                 'However the new layer ' + layer.name +
                                 ' expects ' + str(len(symbolic_weights)) +
                                 ' weights, but the saved weights have ' +
                                 str(len(weight_values)) +
                                 ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(file_path, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    load_weights_from_hdf5_group_by_name(f, model.layers)

    if hasattr(f, 'close'):
        f.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    img_h = 32
    img_w = 280
    maxlabellength = 15

    ini_epoch = 4
    epochs = 20
    batch_size = 128
    # sample_number = 2822680
    sample_number = 3200000  # 2822680 3520000

    # ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"
    MODEL_PATH = "/media/yons/data/dataset/models/text_detection_models/chinese_ocr/models/"
    # modelPath = os.path.join(MODEL_PATH, 'pretrain_model/weights_densenet.h5')
    modelPath = os.path.join(MODEL_PATH, 'output/weights_densenet-04-0.83.h5')
    # modelPath = ""

    char_set = open('char_std_5072.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)
    print("nclass : {}".format(nclass))

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    if os.path.exists(modelPath):
        print("Loading model weights from {}".format(modelPath))
        try:
            basemodel.load_weights(modelPath)
        except Exception as e:
            print(e)
            load_models(file_path=modelPath, model=basemodel)
        print('done!')

    train_loader = gen(ROOT_PATH + 'syn_train3.txt', ROOT_PATH + 'syn_images3', batchsize=batch_size,
                       maxlabellength=maxlabellength,
                       imagesize=(img_h, img_w))
    test_loader = gen(ROOT_PATH + 'syn_test3.txt', ROOT_PATH + 'syn_images3', batchsize=batch_size,
                      maxlabellength=maxlabellength,
                      imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath=MODEL_PATH + 'output/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(epochs)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TensorBoard(log_dir=MODEL_PATH + 'logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=sample_number // batch_size,
                        epochs=epochs,
                        initial_epoch=ini_epoch,
                        validation_data=test_loader,
                        validation_steps=sample_number // (batch_size * 100),
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])
