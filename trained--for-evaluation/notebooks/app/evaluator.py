import os
from glob import glob

from plotter import Plotter
from utils import read_label
import tensorflow as tf
import nltk
from nltk.metrics import distance


def cir_word(hp, gt):
    return distance.edit_distance(hp, gt) / len(gt)


def cir_line(expected, predicted):
    return tf.reduce_mean([cir_word(e, p) for (e, p) in zip(expected, predicted)]).numpy()


def cir_set(labels, result, _len=None):
    if _len is None:
        _len = len(labels)
    return tf.reduce_mean([cir_line(e[:_len], p[:_len]) for (e, p) in zip(labels, result)]).numpy()


class Evaluator:
    def __init__(self, model):
        self._len = 4
        self.model = model
        self.plotter = Plotter( model)

    @staticmethod
    def load_from_path(path, max=None):
        test_images = glob(os.path.join(path, 'images/*.jpg'))
        test_images.sort()

        test_labels_files = glob(os.path.join(path, 'labels/*.pgn'))
        test_labels_files.sort()
        test_labels = [read_label(f) for f in test_labels_files]
        # test_labels= [cleanup( x).lower() for x in test_labels]
        test_labels = [label.split() for label in test_labels]
        if max is None:
            return test_images, test_labels
        else:
            return test_images[:max], test_labels[:max]

    @staticmethod
    def load_test(dataset='test'):
        return Evaluator.load_from_path('../test-data/' + dataset, max=None)

    def evaluate_test_data(self, dataset='test', plot_attention=False):
        result_acc = []

        ac, predicted, expected = self.evaluate_all_data(*Evaluator.load_test(dataset), self._len,
                                                         plot_attention=plot_attention)
        result_acc.append((ac, 'test'))

    def evaluate_all_data(self, images, labels, maxlen, no_teach=True, show_all=False, plot_attention= False):
        result_ac = []
        result = []
        print('evaluating total images: ', len(images), '...')
        for i in range(0, len(images)):
            if i % 100 == 0:
                print('evaluating ', i, '...')
            r, attention_plot, _ = self.model.steps.evaluate(images[i], maxlen, no_teach)
            result.append(r)

            # habilitar para exibir resultado e esperado
            if i < 5 or show_all:
                print('------------------------', i, '------------------------------')
                print('predicted', r)
                print('expected', labels[i])

            if plot_attention:
                # habilitar para plotar attention
                self.plotter.plot_attention(images[i], r, attention_plot, labels[i])

        # calcula a acuracia para cada tamanho
        for _len in range(1, maxlen + 1):
            m = tf.keras.metrics.Accuracy()

            # acuracia para cada teste, até o tamanho atual
            for i in range(0, len(result)):
                useLen = min(len(labels[i]), len(result[i]), _len)
                m.update_state(
                    self.model.tokenizer.texts_to_sequences(labels[i])[:useLen],
                    self.model.tokenizer.texts_to_sequences(result[i])[:useLen])
                # [tokenizer.word_index[w] if w in tokenizer.word_index else 0 for w in labels[i]][:uselen],
                # [tokenizer.word_index[w] if w in tokenizer.word_index else 0  for w in result[i]][:uselen])

            print('len', _len, 'accuracy', float(m.result()), 'cir', cir_set(labels, result, _len))
            result_ac.append(float(m.result()))

        predicted = result
        return result_ac, predicted, labels
