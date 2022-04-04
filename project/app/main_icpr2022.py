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
    NUM_LINES = 8
    train_name = "train_20220404_icpr2022_100k_plus50k_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.0, 1.0, # target_loss, target_acc
                                       (1, 50),  # min_max_epoch
                                       sampled = False,
                                       # use_sample=(0.01, 0.01),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)

if __name__ == '__main__':
    train_()
    print( 'done!')

