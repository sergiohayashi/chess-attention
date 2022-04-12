from glob import glob

import config
from dataset_builder import DatasetGenerator
from model_controller import ModelPredictController, ModelTrainController
from config import TRAIN_FOLDER


def train_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/sequencias-reais-8linhas--70K-.zip',
    ]
    NUM_LINES = 8
    train_name = "train_20220407_icpr2022_ref70k_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.95,  # target_loss, target_acc
                                       (1, 30),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_part2_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/sequencias-reais-8linhas--70K-.zip',
    ]
    NUM_LINES = 8
    train_name = "train_20220407_icpr2022_ref70k_part_2_try_2_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_ref70k_')
    # print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.0001, 0.9999,  # target_loss, target_acc
                                       (1, 40),  # min_max_epoch
                                       # sampled=True,
                                       # use_sample=(0.02, 0.02),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_5k_part1():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()

    model.restoreFromCheckpointName('train_20220407_icpr2022_ref70k_part_2_try_2_--sequencias-reais-8linhas--70K-_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_20220407_icpr2022_70k_ref_5k_part1_try2",
                                       niveis, 0.0001, 0.9999,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_hand2k_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2388.zip',
    ]
    NUM_LINES = 8

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()

    model.restoreFromCheckpointName('train_20220407_icpr2022_70k_ref_5k_part1_try2---8linhas-handwritten--5k_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_20220407_icpr2022_70k_ref_hand2k_",
                                       niveis, 0.0001, 0.9999,
                                       (1, 20),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_hand2k_refTorneios_try2_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/torneios-except-test--only.zip',
    ]
    NUM_LINES = 8

    config.LEARNING_RATE = 0.0001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()

    model.restoreFromCheckpointName('train_20220407_icpr2022_70k_ref_hand2k_---8linhas-handwritten-only-2388_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_20220407_icpr2022_70k_ref_hand2k_refTorneios_try2_",
                                       niveis, 0.0001, 0.9999,
                                       (1, 100),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_hand2k_refTorneios_part2_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/torneios-except-test--only.zip',
    ]
    NUM_LINES = 8

    config.LEARNING_RATE = 0.0001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()

    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_hand2k_refTorneios_try2_--torneios-except-test--only_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_20220407_icpr2022_70k_ref_hand2k_refTorneios_part2_",
                                       niveis, 0.0001, 0.9999,
                                       (1, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )

    # ==> acurácia chega a 100%, mas teste para em 73%


def train_20220407_icpr2022_70k_ref_10k_part1():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten--10k--v20220408-.zip',
    ]
    NUM_LINES = 8

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()

    model.restoreFromCheckpointName('train_20220407_icpr2022_ref70k_part_2_try_2_--sequencias-reais-8linhas--70K-_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum("train_20220407_icpr2022_70k_ref_10k_part1",
                                       niveis, 0.0001, 0.9999,
                                       (1, 50),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_70K_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/sequencias-reais-8linhas--70K-.zip',
    ]
    NUM_LINES = 8

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_part1---8linhas-handwritten--10k--v20220408-_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_70K_',
                                       niveis, 0.0001, 0.9999,  # target_loss, target_acc
                                       (1, 70),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_ref_2k_try2():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2388.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.0001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_part1---8linhas-handwritten--10k--v20220408-_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_try2',
                                       niveis, 0.0001, 0.9999,  # target_loss, target_acc
                                       (1, 70),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_1():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2388.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.00001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_ref_2k_try2---8linhas-handwritten-only-2388_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_1',
                                       niveis, 0.000001, 0.999999,  # target_loss, target_acc
                                       (1, 70),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )

    #   => 0.7927631735801697


def train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_2():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2388.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.00001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_1---8linhas-handwritten-only-2388_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_2',
                                       niveis, 0.00000001, 0.99999999,  # target_loss, target_acc
                                       (1, 500),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_ref_2k_torneio_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/torneios-except-test--only.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.000001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_1---8linhas-handwritten-only-2388_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_torneio_',
                                       niveis, 0.0, 1.0,  # target_loss, target_acc
                                       (1, 100),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2292-nounknown.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.0001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_ref_2k_part2_1---8linhas-handwritten-only-2388_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_',
                                       niveis, 0.0, 1.0,  # target_loss, target_acc
                                       (1, 30),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_pt2_():
    lens = [16]
    niveis = [
        TRAIN_FOLDER + '/dataset/-8linhas-handwritten-only-2292-nounknown.zip',
    ]
    NUM_LINES = 8
    config.LEARNING_RATE = 0.0001

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_---8linhas-handwritten-only-2292-nounknown_best')
    print('acc em teste, antes do treinamento', model.evaluateForTest('test-8lines', _len=16))

    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum('train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_pt2_',
                                       niveis, 0.0, 1.0,  # target_loss, target_acc
                                       (1, 50),  # min_max_epoch
                                       sampled=False,
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


if __name__ == '__main__':
    train_20220407_icpr2022_70k_ref_10k_ref_2k_5k_2292_pt2_()
    print('done!')
