from glob import glob

import config
from dataset_builder import DatasetGenerator
from model_controller import ModelPredictController, ModelTrainController
from config import TRAIN_FOLDER


def train_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/sequencias-reais-8linhas--100K-.zip',
    ]
    train_name = "train_20220404_icpr2022_100k_plus50k_"

    model = ModelTrainController()
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.0, 1.0,  # target_loss, target_acc
                                       (1, 50),  # min_max_epoch
                                       sampled=False,
                                       # use_sample=(0.01, 0.01),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_10K_2Kshuffle():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--10k-shuffle-2k.zip',
    ]
    train_name = "train_20220404_icpr2022_10K_plus2Kshuffle-try3-"

    model = ModelTrainController()
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.0, 1.0,  # target_loss, target_acc
                                       (1, 20),  # min_max_epoch
                                       sampled=False,
                                       # use_sample=(0.01, 0.01),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_icpf22_handwritten_5k_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_20220405_icpr22_handwritten_5k_try2_"

    model = ModelTrainController()
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.20, 0.90,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_icpf22_handwritten_5k_att100_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_20220405_icpr22_handwritten_5k_att100_try4_"
    config.force_size_mode(1)

    model = ModelTrainController()
    model.load()
    model.initTrainSession(BATCH_SIZE=4)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.20, 0.90,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_icpf22_handwritten_5k_att100_part2():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    train_name = "train_20220405_icpr22_handwritten_5k_att100_part_2"
    config.force_size_mode(1)

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName(
        'train_20220405_icpr22_handwritten_5k_att100_try4_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=4)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.20, 0.90,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_icpf22_handwritten_100K_ref_5k_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_20220404_icpr2022_100k_plus50k_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=4)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_try2_",
                                       niveis, 0.20, 0.90,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_icpf22_handwritten_100K_ref_5k_part2_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_icpf22_handwritten_100K_ref_5k_try2_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_part2_",
                                       niveis, 0.01, 0.98,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_icpf22_handwritten_100K_ref_5k_part3_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_icpf22_handwritten_100K_ref_5k_part2_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_part3_",
                                       niveis, 0.01, 0.999,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_icpf22_handwritten_100K_ref_5k_part4_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_icpf22_handwritten_100K_ref_5k_part3_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_part4_",
                                       niveis, 0.0, 1.0,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_icpf22_handwritten_100K_ref_5k_part5_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_icpf22_handwritten_100K_ref_5k_part4_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_part5_",
                                       niveis, 0.0, 1.0,
                                       (1, 50),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_icpf22_handwritten_100K_ref_5k_part6_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]

    model = ModelTrainController()
    model.load()

    model.restoreFromCheckpointName('train_icpf22_handwritten_100K_ref_5k_part5_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_icpf22_handwritten_100K_ref_5k_part6_",
                                       niveis, 0.0, 1.0,
                                       (1, 50),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


if __name__ == '__main__':
    train_icpf22_handwritten_100K_ref_5k_part6_()
    print('done!')
