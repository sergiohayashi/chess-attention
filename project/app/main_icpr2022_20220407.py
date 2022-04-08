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


if __name__ == '__main__':
    train_part2_()
    print('done!')
