import os

ROOT_DIR = os.path.abspath('../..')
TRAIN_FOLDER = os.path.join(ROOT_DIR, 'train-folder')
VOCAB_FOLDER = os.path.join(ROOT_DIR, 'vocab')

USE_BIG_PLOT = False
PRINT_EACH_MOVE = False

STOP_FILE = os.path.join(ROOT_DIR, 'stop')
print('Para parar o treinamento, criar o arquivo ', STOP_FILE)

predefined_size_modes = [
    {  # 0: default
        "input_shape": (800, 862),
        "attention_shape": (50, 53),
        "vgg_layer": -2
    }, {  # 1: com tamanho na cnn maior
        "input_shape": (800, 862),
        "attention_shape": (100, 107),
        "vgg_layer": -6
    }, {  # 2
        "input_shape": (400, 430),
        "attention_shape": (25, 26),
        "vgg_layer": -2
    }
]

size_mode = predefined_size_modes[0]
print('Using default size mode: ', size_mode)

def force_size_mode(id):
    global size_mode
    size_mode = predefined_size_modes[id]
    print('Forced size mode: ', size_mode)

SAVE_IF_BETTER_THAN = 0.6
SAVE_INCREMENT = 0.001

