import sys

from config import TRAIN_FOLDER
from model_controller import ModelTrainController
import tensorflow as tf


def train_(n_train, n_valid):
    lens = [4]  # <= tamanho 4!
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_icpr2022_reply_reference-len4_" + str(n_train)

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
    if len( sys.argv)< 3:
        print( 'Erro! Esperado dois parametros, n_train, n_valid')
        sys.exit(0)

    n_train = int( sys.argv[1])
    n_valid = int( sys.argv[2])
    print( 'n_train', n_train, 'n_valid', n_valid)
    train_(n_train, n_valid)

    print('done!')
