import os
import shutil
import uuid
from glob import glob
import random
from pathlib import Path

import numpy as np

import config
from game_data import GameData
from img_utils import resize
from io_utils import write_image, write_label, load_image_
from vocab import load_vocab_175


class FixPartsLib:
    def __init__(self):
        pass

    def fix(self):
        w = set()
        files = glob(os.path.join(config.ROOT_DIR + '/parts-lib/*/*/*.jpg'))

        for f in files:
            p = os.path.normpath(f).split(os.sep)
            w_path = p[-2]
            w_name = p[-1].split('_')[0]
            if w_path != w_name:
                print( f)
                w.add( w_name)

                shutil.move( f, f.replace( w_path, w_name))
                print( f, '-> ', f.replace( w_path, w_name))

        print( w)

if __name__ == "__main__":
    FixPartsLib().fix()
