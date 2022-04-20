import os
import shutil
import uuid
from glob import glob
import random
from pathlib import Path

import numpy as np

import config
import io_utils
from game_data import GameData
from img_utils import resize
from io_utils import write_image, write_label, load_image_
from vocab import load_vocab_175


class FiltraUnknown:
    def __init__(self):
        pass

    def fix(self, path):
        count = 0
        words = set(load_vocab_175())
        print(words)

        badfiles = []
        files = glob(os.path.join(path + '/*/labels/*.pgn'))
        print(len(files))
        for f in files:
            label_set = set(io_utils.read_label(f).split())
            if not label_set.issubset(words):
                print('{} {}'.format(Path(f).name, label_set - words))
                count = count+1
                badfiles.extend( glob(os.path.join(path + '/*/*/{}.*'.format( Path(f).name.replace( '.pgn', '')))))
            else:
                pass
                # print('{} OK'.format(Path(f).name))
        print( 'Found {}'.format( count))

        print( '<< DELETE >>')
        for f in badfiles:
            print( f)
            os.remove( f)

if __name__ == "__main__":
    FiltraUnknown().fix(
        'C:\mestrado/repos-github/chess-attention/train-folder/tmp/-8linhas-handwritten--10k--v20220408--nounk')
