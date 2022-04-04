import os

ROOT_DIR = os.path.abspath('../..')
TRAIN_FOLDER = os.path.join(ROOT_DIR, 'train-folder')
VOCAB_FOLDER = os.path.join(ROOT_DIR, 'vocab')

USE_BIG_PLOT = False
FORCE_INPUT_SIZE = False
FORCED_INPUT_SIZE = {
    "INPUT_SHAPE": (400, 430),
    "ATTENTION_SHAPE": (25, 26),
}
PRINT_EACH_MOVE= False
