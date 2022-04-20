from config import TRAIN_FOLDER
from model_controller import ModelTrainController
import tensorflow as tf


def train_icpr2022_reply_reference_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_icpr2022_reply_reference_"

    model = ModelTrainController()
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       sampled=True,
                                       use_sample=(5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_(n_train, n_valid):
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_icpr2022_reply_reference_" + str(n_train)

    model = ModelTrainController()
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 50),
                                       sampled=True,
                                       use_sample=(n_train, n_valid),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


if __name__ == '__main__':
    # train_icpr2022_reply_reference_();
    train_(4500, 400)   # <= TODO!
    # train_(4000, 400)
    # train_(3500, 400)
    # train_(3000, 400)
    # train_(2500, 400)
    # train_(2000, 400)
    # train_(1500, 400)
    # train_(1000, 400)
    # train_(500, 50)
    # train_(200, 50)
    # train_(50, 50)

    print('done!')
