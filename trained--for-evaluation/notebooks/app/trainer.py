import tensorflow as tf
import utils
import numpy as np
import os
import time
from glob import glob
import random
from pathlib import Path

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))
print(tf.__version__)


class TrainerController:

    def __init__(self, model):
        self.model = model
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 1000
        self.train_dataset = None
        self.valid_dataset = None
        self.train_num_steps = None
        self.valid_num_steps = None

        self.loss_plot_train = []
        self.loss_plot_valid = []
        self.acc_plot_train = []
        self.acc_plot_valid = []

        self._epoch = 0

    def prepareFilesForTrain(self, folder):
        # load test data
        print('loading train data...');
        train_img_names, train_label_indexes, train_label_words = DataHelper.load_data_from(
            folder + '/train', self.model.tokenizer, True)  # '/content/dataset-v035--2lines-32k-v5.1.1/train')
        print('loading valid data...');
        valid_img_names, valid_label_indexes, valid_label_words = DataHelper.load_data_from(
            folder + '/valid', self.model.tokenizer, True)  # '/content/dataset-v035--2lines-32k-v5.1.1/valid')

        # build cache
        print('building cache for train data...');
        DataHelper.build_cache_for(self.model, train_img_names)
        print('building cache for valid data...');
        DataHelper.build_cache_for(self.model, valid_img_names)

        # build dataset
        print('building final dataset...');
        self.train_dataset = DataHelper.build_dataset(self.model, train_img_names, train_label_indexes,
                                                      self.BUFFER_SIZE, self.BATCH_SIZE)
        self.valid_dataset = DataHelper.build_dataset(self.model, valid_img_names, valid_label_indexes,
                                                      self.BUFFER_SIZE, self.BATCH_SIZE)

        self.train_num_steps = len(train_img_names) // self.BATCH_SIZE
        self.valid_num_steps = len(valid_img_names) // self.BATCH_SIZE
        print('building final dataset done');

    def trainUntil(self, target_loss, max_epoch):
        self.train_more(max_epoch, target_loss,
                        self.train_dataset, self.valid_dataset, self.train_num_steps, self.valid_num_steps)

    def train_more(self, MAX_EPOCH, loss_target, train_dataset, valid_dataset,
                   train_num_steps, valid_num_steps,
                   train_length=4, val_loss_limit=0):  # , n_epoch):

        print("-- loss_target=>", loss_target, " train_length=", train_length)
        for _ in range(0, MAX_EPOCH):
            self._epoch += 1
            start = time.time()
            total_loss = 0

            self.train_acc_metric.reset_states()

            #
            # training loop
            #
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.model.steps.train_step(img_tensor, target, train_length)
                total_loss += t_loss

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        self._epoch, batch, batch_loss.numpy() / train_length))

            train_loss = total_loss / train_num_steps
            self.loss_plot_train.append(train_loss)
            train_acc = float(self.train_acc_metric.result())
            self.acc_plot_train.append(train_acc)

            #
            # validation loop
            #
            valid_total_loss = 0
            for (batch, (img_tensor, target)) in enumerate(valid_dataset):
                batch_loss, t_loss = self.model.steps.test_step(img_tensor, target, train_length)
                valid_total_loss += t_loss
            valid_loss = valid_total_loss / valid_num_steps
            self.loss_plot_valid.append(valid_loss)
            valid_acc = float(self.valid_acc_metric.result())
            self.acc_plot_valid.append(valid_acc)

            #
            # print..
            #

            print('Epoch {} Loss {:.6f}  acc: {:.4f} [ Validation Loss {:.6f} valid_acc: {:.4f} ]'.format(
                self._epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            # print_time()

            #
            # target reached?
            #
            if loss_target > 0 and (train_loss) <= loss_target:
                print("Target reached! stop!", ' len= ', train_length)
                return True

        print('epoch exceeded')
        return False


class DataHelper:
    @staticmethod
    def load_data_from(path, tokenizer, SAMPLED):
        print('loading data from ', path)
        image_files = glob(os.path.join(path, 'images/*.jpg'))
        image_files.sort()

        label_files = glob(os.path.join(path, 'labels/*.pgn'))
        label_files.sort()
        labels = [utils.read_label(f) for f in label_files]
        # labels= [cleanup( x).lower() for x in labels]
        labels = ['<start> ' + label + ' <end>' for label in labels]

        # poderia ser menor... mas pega os primeiros 10. Nem precisava restringir...
        labels = [label.split()[0:16 + 1] for label in labels]

        # somente uma parte por enquanto
        if SAMPLED:
            n = int(len(image_files) * 0.10)
            combined = list(zip(image_files, labels))
            random.Random(0).shuffle(combined)
            image_files[:], labels[:] = zip(*combined[:n])
            print("SAMPLED!!  size= ", len(image_files), len(labels))

        label_indexes = tokenizer.texts_to_sequences(labels)
        for i in range(0, 3):
            print(labels[i], '=>', label_indexes[i])

        return image_files, label_indexes, labels

    @staticmethod
    def build_cache_for(model, image_files_list):
        def not_exists(f):
            return not Path(f).is_file()

        # filtra imagens com cache ja criado..
        print('before', len(image_files_list))
        image_files_list = [f for f in image_files_list if not_exists(f + ".npy")]
        print('after', len(image_files_list))

        if len(image_files_list) <= 0:
            return

        encode_train = sorted(set(image_files_list))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            model.steps.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8)  # (16)

        for img, path in image_dataset:
            batch_features = model.image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

    @staticmethod
    def build_dataset(model, img_names, label_indexes, BUFFER_SIZE, BATCH_SIZE):
        def map_func(img_name, cap):
            img_tensor = np.load(img_name.decode('utf-8') + '.npy')
            return img_tensor, cap

        # deixa no mesmo tamanho, maximo 32
        label_indexes = [label[:32] for label in label_indexes]
        label_indexes = tf.keras.preprocessing.sequence.pad_sequences(label_indexes, padding='post')

        dataset = tf.data.Dataset.from_tensor_slices((img_names, label_indexes))

        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(BUFFER_SIZE, seed=0).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
