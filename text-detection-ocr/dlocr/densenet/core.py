import json
from concurrent.futures import ThreadPoolExecutor

import keras.backend as K
import numpy as np
from PIL import Image
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, AveragePooling2D, ZeroPadding2D, Permute, \
    TimeDistributed, Flatten, Dense, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.training_utils import multi_gpu_model

import h5py
from keras.engine.topology import preprocess_weights_for_loading
from dlocr.densenet.data_loader import DataLoader


def _dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = _conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb])
        nb_filter += growth_rate
    return x, nb_filter


def _conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def _transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)

    return x, nb_filter


def _ctc_loss(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def single_img_process(img):
    im = img.convert('L')
    scale = im.size[1] * 1.0 / 32
    w = im.size[0] / scale
    w = int(w)

    im = im.resize((w, 32), Image.ANTIALIAS)
    img = np.array(im).astype(np.float32) / 255.0 - 0.5
    img = img.reshape((32, w, 1))
    return img


def pad_img(img, len, value):
    out = np.ones(shape=(32, len, 1)) * value
    out[:, :img.shape[1], :] = img
    return out


def process_imgs(imgs):
    tmp = []
    with ThreadPoolExecutor() as executor:
        for img in executor.map(single_img_process, imgs):
            tmp.append(img)

    max_len = max([img.shape[1] for img in tmp])

    output = []
    with ThreadPoolExecutor() as executor:
        for img in executor.map(lambda img: pad_img(img, max_len, 0.5), tmp):
            output.append(img)

    return np.array(output)


def decode_single_line(pred_text, nclass, id_to_char):
    char_list = []

    # pred_text = pred_text[np.where(pred_text != nclass - 1)[0]]

    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (pred_text[i] != pred_text[i - 1]) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(id_to_char[pred_text[i]])
    return u''.join(char_list)


def decode(pred, nclass, id_to_char):
    lines = []

    pred_texts = pred.argmax(axis=2)

    with ThreadPoolExecutor() as executor:
        for line in executor.map(lambda pred_text: decode_single_line(pred_text, nclass, id_to_char), pred_texts):
            lines.append(line)

    return lines


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
            if weight_names == ['out/kernel:0', 'out/bias:0']:
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


class DenseNetOCR:

    def __init__(self,
                 num_classes,
                 lr=0.0005,
                 image_height=32,
                 image_channels=1,
                 maxlen=50,
                 dropout_rate=0.2,
                 weight_decay=1e-4,
                 filters=64,
                 weight_path=None,
                 num_gpu=1):
        self.image_shape = (image_height, None, image_channels)
        self.lr = lr
        self.image_height, self.image_channels = image_height, image_channels
        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.filters = filters
        self.num_classes = num_classes
        self.num_gpu = num_gpu
        self.base_model, self.model, self.parallel_model = self.__build_model()
        if weight_path is not None:
            print("Loading model weights from {}".format(weight_path))
            try:
                self.base_model.load_weights(weight_path)
            except Exception as e:
                print(e)
                load_models(file_path=weight_path, model=self.base_model)
            print('done!')

    def config(self):
        return {
            "lr": self.lr,
            "num_classes": self.num_classes,
            "image_height": self.image_height,
            "image_channels": self.image_channels,
            "maxlen": self.maxlen,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "filters": self.filters
        }

    def __build_model(self):
        input = Input(shape=self.image_shape, name="the_input")
        nb_filter = self.filters

        x = Conv2D(nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
                   use_bias=False, kernel_regularizer=l2(self.weight_decay))(input)

        # 64 +  8 * 8 = 128
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)
        # 128
        x, nb_filter = _transition_block(x, 128, self.dropout_rate, 2, self.weight_decay)

        # 128 + 8 * 8 = 192
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)
        # 192->128
        x, nb_filter = _transition_block(x, 128, self.dropout_rate, 2, self.weight_decay)

        # 128 + 8 * 8 = 192
        x, nb_filter = _dense_block(x, 8, nb_filter, 8, None, self.weight_decay)

        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

        x = Permute((2, 1, 3), name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        y_pred = Dense(self.num_classes, name='out', activation='softmax')(x)

        base_model = Model(inputs=input, outputs=y_pred)

        labels = Input(shape=(self.maxlen,), dtype='float32', name="the_labels")
        input_length = Input(shape=(1,), name="input_length", dtype='int64')
        label_length = Input(shape=(1,), name="label_length", dtype='int64')

        loss_out = Lambda(_ctc_loss, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

        model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        adam = Adam(self.lr)
        parallel_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

        return base_model, model, parallel_model

    def train(self, epochs, train_data_loader: DataLoader, valid_data_loader: DataLoader, **kwargs):
        self.parallel_model.fit_generator(generator=train_data_loader.load_data(), epochs=epochs,
                                          steps_per_epoch=train_data_loader.steps_per_epoch,
                                          validation_data=valid_data_loader.load_data(),
                                          validation_steps=valid_data_loader.steps_per_epoch,
                                          **kwargs)

    def predict(self, image, id_to_char):
        if type(image) == str:
            img = Image.open(image)
        else:
            img = image
        im = img.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)

        im = im.resize((w, 32), Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])

        y_pred = self.base_model.predict(X)

        y_pred = y_pred.argmax(axis=2)
        out = decode_single_line(y_pred[0], self.num_classes, id_to_char)

        return out, im

    def predict_multi(self, images, id_to_char):
        """
        Predict multi images
        :param images: [Image]
        :param id_to_char:
        :return:
        """

        X = process_imgs(images)
        y_pred = self.base_model.predict_on_batch(X)

        return decode(y_pred, self.num_classes, id_to_char)

    @staticmethod
    def save_config(obj, config_path: str):
        with open(config_path, 'w+') as outfile:
            json.dump(obj.config(), outfile)

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, 'r') as infile:
            return dict(json.load(infile))
