import os
import shutil
from glob import glob
from pathlib import Path
import random

import config


def generate_mosaic_dataset(name, n):
    # cria diretorio target
    root = config.TRAIN_ROOT_DIR + '/tmp/' + name
    print( 'path: ', root)
    Path(os.path.join(root, 'train/images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, 'train/labels')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, 'valid/images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, 'valid/labels')).mkdir(parents=True, exist_ok=True)

    pool_dir = config.BUILDER_ROOT_DIR + '/pool_8lines/'  # pool de imagens
    files = []
    hand_files = glob(os.path.join(pool_dir, 'hand', 'labels/*.pgn'))
    print( 'hand files: ', len( hand_files))
    files.extend( hand_files)

    torneio_files = glob(os.path.join(pool_dir, 'torneio-except-test', 'labels/*.pgn'))
    print( 'torneios: ', len( torneio_files))
    files.extend( torneio_files)

    n_mosaic = n - len(files)
    files.extend(glob(os.path.join(pool_dir, 'mosaic', 'labels/*.pgn'))[:n_mosaic])
    print( 'mosaic: ', n_mosaic)

    N_train = int(len(files) * .8)
    print('train, valid', N_train, len(files) - N_train)

    # shuffle
    random.shuffle(files)
    print('total', len(files))

    # copia
    for i in range(0, len(files)):
        if (i % 1000 == 0):
            print(i, '....')

        f = files[i]
        folder = 'train' if i < N_train else 'valid'

        # copia
        dest_file = os.path.join(root, folder, 'labels', Path(f).name)
        #         print( dest_file)
        shutil.copyfile(f, dest_file)
        shutil.copyfile(
            f.replace('labels', 'images').replace('pgn', 'jpg'),
            dest_file.replace('labels', 'images').replace('pgn', 'jpg'))


if __name__ == "__main__":
    generate_mosaic_dataset('sequencias-reais-8linhas--', 50000)
    print('done!')
