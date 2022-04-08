import os
import shutil
import uuid
from glob import glob
import random
from pathlib import Path

import cv2
import imutils as imutils
import numpy as np

import config
from game_data import GameData
from img_utils import resize, image_resize
from io_utils import write_image, write_label, load_image_, read_label_for
from vocab import load_vocab_175


class PgnToJpg:
    def __init__(self):
        pass

    @staticmethod
    def convert():
        files = glob('C:/Users/hayashi/Pictures/Screenshots/*.png')
        for f in files:
            print( f)
            # Load .png image
            image = cv2.imread(f)

            # Save .jpg image
            cv2.imwrite(f.replace('.png', '.jpg'), image)


if __name__ == "__main__":
    PgnToJpg.convert()
