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
from io_utils import write_image, write_label, load_image_
from vocab import load_vocab_175


class PartsGenerator:
    def __init__(self):
        pass

    def generate(self):

        img_base = cv2.imread('C:/mestrado/repos-github/chess-attention/dataset-builder/templates/part.jpg')
        img_base = resize(img_base, 339, 72)

        org_dir = "C:/Users\hayashi/Pictures/Screenshots/jpg/*.jpg"
        dest_dir = "C:/mestrado/repos-github/chess-attention/dataset-builder/parts-lib_work/from-hebraica"

        files = glob(org_dir)
        for f in files:
            p = os.path.normpath(f).split(os.sep)
            # print( p)
            w = p[-1].split('_')[0]
            # print( w, p)

            img = cv2.imread(f)
            img = imutils.resize(img, height=72 - 10)

            img2 = img_base.copy()
            # print( 'img.shape', img.shape)
            dx = (img_base.shape[1] - img.shape[1]) // 2
            img2[10:img.shape[0] + 10, dx:img.shape[1] + dx] = img

            # img = image_resize( img, 339, 72)

            if not os.path.exists(os.path.join(dest_dir, w)):
                print('lance {} nao existe!'.format(w))
                continue

            copy_to_file = os.path.join(dest_dir, w, w + '_' + str(uuid.uuid4())[:8]+'.jpg')
            print(copy_to_file)
            cv2.imwrite(copy_to_file, img2)


if __name__ == "__main__":
    PartsGenerator().generate()
