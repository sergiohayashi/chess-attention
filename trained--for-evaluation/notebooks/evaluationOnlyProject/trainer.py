import tensorflow as tf

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))
import numpy as np
import os
import time
from glob import glob
import random

print(tf.__version__)

class Trainer:

    def __init__(self):
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_plot_train = []
        self.loss_plot_valid = []
        self.acc_plot_train = []
        self.acc_plot_valid = []

    def train(self, model, target):
        # load test data
        train_img_names, train_label_indexes, train_label_words = DataLoader(model.tokenizer).load_data_from(
            '/content/dataset-v035--2lines-32k-v5.1.1/train')
        valid_img_names, valid_label_indexes, valid_label_words = DataLoader(model.tokenizer).load_data_from(
            '/content/dataset-v035--2lines-32k-v5.1.1/valid')

        # build cache
        DataCacheBuilder().build_cache_for(train_img_names)
        DataCacheBuilder().build_cache_for(valid_img_names)

        # build dataset
        train_dataset = DatasetBuilder().build_dataset(train_img_names, train_label_indexes)
        valid_dataset = DatasetBuilder().build_dataset(valid_img_names, valid_label_indexes)

        train_num_steps = len(train_img_names) // BATCH_SIZE
        valid_num_steps = len(valid_img_names) // BATCH_SIZE

        train_dataset, valid_dataset = None, None

        # until target
        # train
        self.train_more(0.01, 200, train_dataset, valid_dataset, train_num_steps, valid_num_steps)

    def train_more(self, MAX_EPOCH, loss_target, train_dataset, valid_dataset,
                   train_num_steps, valid_num_steps,
                   train_length=4, val_loss_limit=0):  # , n_epoch):

        print("-- loss_target=>", loss_target, " train_length=", train_length)
        for _ in range(0, MAX_EPOCH):
            _epoch += 1
            start = time.time()
            total_loss = 0

            self.train_acc_metric.reset_states()

            #
            # training loop
            #
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.model.train_step(img_tensor, target, train_length)
                total_loss += t_loss

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        _epoch, batch, batch_loss.numpy() / train_length))

            train_loss = total_loss / train_num_steps
            self.loss_plot_train.append(train_loss)
            train_acc = float(self.train_acc_metric.result())
            self.acc_plot_train.append(train_acc)

            # #
            # # validation loop
            # #
            # valid_total_loss = 0
            # for (batch, (img_tensor, target)) in enumerate(valid_dataset):
            #     batch_loss, t_loss = test_step(img_tensor, target, train_length)
            #     valid_total_loss += t_loss
            # valid_loss = valid_total_loss / valid_num_steps
            # loss_plot_valid.append(valid_loss)
            # valid_acc = float(valid_acc_metric.result())
            # acc_plot_valid.append(valid_acc)
            #
            # #
            # # print..
            # #

            print('Epoch {} Loss {:.6f}  acc: {:.4f} [ Validation Loss {:.6f} valid_acc: {:.4f} ]'.format(
                _epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            print_time()

            #
            # target reached?
            #
            if loss_target > 0 and (train_loss) <= loss_target:
                print("Target reached! stop!", ' len= ', train_length)
                return True

        print('epoch exceeded')
        return False


class DataLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer;

    def load_data_from(self, path):
        image_files = glob(os.path.join(path, 'images/*.jpg'))
        image_files.sort()

        label_files = glob(os.path.join(path, 'labels/*.pgn'))
        label_files.sort()
        labels = [read_label(f) for f in label_files]
        # labels= [cleanup( x).lower() for x in labels]
        labels = ['<start> ' + label + ' <end>' for label in labels]

        # poderia ser menor... mas pega os primeiros 10. Nem precisava restringir...
        labels = [label.split()[0:16 + 1] for label in labels]

        # somente uma parte por enquanto
        if SAMPLED:
            n = int(len(image_files) * 0.50)
            combined = list(zip(image_files, labels))
            random.Random(0).shuffle(combined)
            image_files[:], labels[:] = zip(*combined[:n])
            print("SAMPLED!!  size= ", len(image_files), len(labels))

        label_indexes = self.tokenizer.texts_to_sequences(labels)
        for i in range(0, 3):
            print(labels[i], '=>', label_indexes[i])

        return image_files, label_indexes, labels

    def load_from(self):
        # shuffle
        # train_img_names, train_label_indexes, train_label_words= load_data_from( '/content/shuffle_8lines_32K/train')
        # valid_img_names, valid_label_indexes, valid_label_words= load_data_from( '/content/shuffle_8lines_32K/valid')

        train_img_names, train_label_indexes, train_label_words = self.load_data_from(
            '/content/dataset-v035--2lines-32k-v5.1.1/train')
        valid_img_names, valid_label_indexes, valid_label_words = self.load_data_from(
            '/content/dataset-v035--2lines-32k-v5.1.1/valid')


class DataCacheBuilder:
    def __init__(self):
        pass

    # def load_image(image_path):
    #     img = tf.io.read_file(image_path)
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.image.resize(img, (200, 862))  # (450, 339))  #original=(576, 678, 3)
    #     # img = tf.image.resize(img, (540, 407)) #(450, 339))  #original=(576, 678, 3)
    #     # img = tf.keras.applications.inception_v3.preprocess_input(img)
    #     img = tf.keras.applications.vgg19.preprocess_input(img)
    #     return img, image_path

    def build_cache_for(self, img_name_vector):
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            self.model.steps.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8)  # (16)

        for img, path in image_dataset:
            batch_features = self.image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())
    #


class DatasetBuilder:
    def __init__(self):
        pass

    def build_dataset(self, img_names, label_indexes):
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
