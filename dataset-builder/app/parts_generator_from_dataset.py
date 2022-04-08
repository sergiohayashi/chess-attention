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


class PartsGeneratorFromLimpos:
    def __init__(self):
        self.img_base = cv2.imread('C:/mestrado/repos-github/chess-attention/dataset-builder/templates/part.jpg')
        self.img_base = resize(self.img_base, 339, 72)


    def fix_ratio(self, img):

        # org_dir = "C:/Users\hayashi/Pictures/Screenshots/jpg/*.jpg"
        # dest_dir = "C:/mestrado/repos-github/chess-attention/dataset-builder/parts-lib_work/from-hebraica"
        #
        # files = glob(org_dir)
        # for f in files:
        #     p = os.path.normpath(f).split(os.sep)
        #     # print( p)
        #     w = p[-1].split('_')[0]
        #     # print( w, p)
        #
        #     img = cv2.imread(f)
        img = imutils.resize(img, height=72 - 10)

        img2 = self.img_base.copy()
        # print( 'img.shape', img.shape)
        dx = (self.img_base.shape[1] - img.shape[1]) // 2
        img2[10:img.shape[0] + 10, dx:img.shape[1] + dx] = img
        return img2

        # img = image_resize( img, 339, 72)

        # if not os.path.exists(os.path.join(dest_dir, w)):
        #     print('lance {} nao existe!'.format(w))
        #     continue
        #
        # copy_to_file = os.path.join(dest_dir, w, w + '_' + str(uuid.uuid4())[:8]+'.jpg')
        # print(copy_to_file)
        # cv2.imwrite(copy_to_file, img2)


    def generate_from_hebraica(self):
        images_path = config.BUILDER_ROOT_DIR+ '/originais/hebraica/images'

        files = glob( os.path.join( images_path, '*.jpg'))
        for f in files:
            print( f)

            labels = read_label_for( f).split()

            img = cv2.imread(f)

            print( img.shape)
            h = img.shape[0]/8
            w = img.shape[1]/2
            dx = 10
            dy = 0

            for i in range( 0, 8):
                for j in range( 0, 2):

                    cut = img[
                          max( 0, int(i*h))+ dy:int(h*(i+1))- 2*dy,
                          max( 0, int(j*w))+ dy:int(w*(j+1))- 2*dy]
                    print( cut.shape)
                    label = labels[j*2+i]
                    fname = label+ '_'+ str(uuid.uuid4())[:8]
                    cv2.imwrite('C:/Users/hayashi/Pictures/tmp/{}.jpg'.format( fname), self.fix_ratio( cut))

            return



    def generate_from_carnaval(self):
        images_path = config.BUILDER_ROOT_DIR+ '/pool_8lines/torneio-except-test/images'
        dest_dir = "C:/mestrado/repos-github/chess-attention/dataset-builder/parts-lib_tmp/from-carnaval-except-test"

        files = glob( os.path.join( images_path, '*.jpg'))
        for f in files:
            print( f)
            labels = read_label_for( f).split()
            img = cv2.imread(f)

            h = img.shape[0]/8
            w = img.shape[1]/2
            dx = 0
            dy = 10

            for i in range( 0, 8):
                for j in range( 0, 2):
                    cut = img[
                          max( 0, int(i*h)):int(h*(i+1)),
                          max( 0, int(j*w)):int(w*(j+1))+ 10]
                    word = labels[i*2+j]
                    if not os.path.exists(os.path.join(dest_dir, word)):
                        print('lance {} nao existe!'.format(word))
                        continue

                    # fname = w+ '_'+ str(uuid.uuid4())[:8]
                    copy_to_file = os.path.join(dest_dir, word, word + '_' + str(uuid.uuid4())[:8] + '.jpg')
                    cv2.imwrite(copy_to_file, cut)

if __name__ == "__main__":
    PartsGeneratorFromLimpos().generate_from_carnaval()

