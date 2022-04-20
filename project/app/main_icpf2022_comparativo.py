from config import TRAIN_FOLDER
from model_controller import ModelTrainController
import tensorflow as tf


def train_(train_name, n_train, n_valid):
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--10k--v20220408--nounk.zip',
    ]

    model = ModelTrainController()
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name + str(n_train),
                                       niveis, 0.1, 0.9,  # target_loss, target_acc
                                       (1, 50),  # min_max_epoch
                                       sampled=True,
                                       use_sample=(n_train, n_valid),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


if __name__ == '__main__':
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 500, 100)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 1000, 100)
    train_('train_icpr2022_comparativo_10k_sampled_5_1_', 2000, 200)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 3000, 300)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 4000, 400)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 5000, 500)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 7000, 700)
    # train_('train_icpr2022_comparativo_10k_sampled_5_', 10000, 1000)
    print('done!')
