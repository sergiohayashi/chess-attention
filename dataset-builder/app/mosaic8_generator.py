import os
import uuid
from glob import glob
import random

import numpy as np

import config
from game_data import GameData
from img_utils import resize
from io_utils import write_image, write_label, load_image_
from vocab import load_vocab_175


class Mosaic8Generator:

    def __init__(self):
        self.words = load_vocab_175();
        print( 'self.words', self.words)

        self.word_cuts = {}
        for w in self.words:
            self.word_cuts[w] = glob(os.path.join(config.ROOT_DIR + '/parts-lib/*', w, "*.jpg"))
        self.verbose= False

    def generate_8lines_mosaic(self, qtd, output_dir=config.ROOT_DIR + '/pool_8lines/mosaic', verbose=False):
        # le jogos
        jogos = GameData.load(qtd, shuffled=True, inwords=self.words)
        if verbose:
            self.verbose = True

        os.makedirs(os.path.join( output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join( output_dir, 'labels'), exist_ok=True)

        # gera imagens
        for i in range(0, len( jogos)):
            if i % 100 == 0:
                print(i)
            self.generate_for_label(jogos[i][:16], output_dir)

    def generate_for_label(self, label, folder, tag=None):
        cuts = []
        for w in label:
            cuts.append(self.get_random_cut_for(w))
        img = self.rebuild(cuts)
        #     show( img)
        # img = resize(img, 345, 300)

        if tag is None:
            tag = 'mosaic'
        fname = tag + "-" + str(uuid.uuid4())[:8]
        #     show( img)
        #     print( " ".join( label))
        write_image(os.path.join(folder, 'images', fname + ".jpg"), img)
        write_label(os.path.join(folder, 'labels', fname + ".pgn"), " ".join(label))
        if self.verbose:
            print( 'generated ', os.path.join(folder, 'images', fname + ".jpg"))

    def rebuild(self, cuts):
        col1 = np.concatenate((
            cuts[0], cuts[2], cuts[4], cuts[6], cuts[8],
            cuts[10], cuts[12], cuts[14]
        ), axis=0)
        col2 = np.concatenate((
            cuts[1], cuts[3], cuts[5], cuts[7], cuts[9],
            cuts[11], cuts[13], cuts[15]
        ), axis=0)
        return np.concatenate((col1, col2), axis=1)

    def get_random_cut_for(self, w):
        f = self.word_cuts[w][random.randrange(0, len(self.word_cuts[w]))]
        if f is None:
            print('Oops nao encontrado para ', w)
            print(self.word_cuts[w])
        #     print( w, f)
        return resize(load_image_(f), 339, 72)


if __name__ == "__main__":
    Mosaic8Generator().generate_8lines_mosaic(100000, verbose=False)

