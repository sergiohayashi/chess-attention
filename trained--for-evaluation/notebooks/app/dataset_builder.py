import os
import os.path
import random
import shutil
from glob import glob
from pathlib import Path


class DatasetGenerator:

    @staticmethod
    def generate_dataset_v2_2lines(ver,
                                   n_hand, n_parts, n_sync, n_shuffle,
                                   n_parts_seq,
                                   n_parts_ec,
                                   n_parts_ec_v2,
                                   include_all_except_test=False, test=True):
        # N_ratio = (parts,syn)

        n_hand = min(2010, n_hand)
        N = n_parts + n_sync + n_hand + n_shuffle + n_parts_seq + n_parts_ec + n_parts_ec_v2
        #     print( 'gerar total', N, 'hand', n_hand, 'parts', n_parts, 'syc', n_sync, 'shuffle', n_shuffle,
        #          'parts_seq', n_parts_seq
        #          )

        # cria diretorio
        root = '../train-folder/tmp/' + ver
        Path(os.path.join(root, 'train/images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, 'train/labels')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, 'valid/images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, 'valid/labels')).mkdir(parents=True, exist_ok=True)

        files = []

        # le os arquivos
        pool_base = '../../../dataset-builder/pool_2lines'
        if n_hand > 0:
            files_hand = glob(pool_base + '/hand/labels/*.pgn')
            print('files_hand', n_hand, '/', len(files_hand));
            files.extend(files_hand[:n_hand])

        if n_sync > 0:
            files_syn = glob(pool_base + '/syn/labels/*.pgn')
            print('files_syn', n_sync, '/', len(files_syn));
            files.extend(files_syn[:n_sync])

        if n_parts > 0:
            files_parts = glob(pool_base + '/parts/labels/*.pgn')
            print('files_parts', n_parts, '/', len(files_parts));
            files.extend(files_parts[:n_parts])

        if n_shuffle > 0:
            files_shuffle = glob(pool_base + '/shuffle/labels/*.pgn')
            print('files_parts', n_shuffle, '/', len(files_parts));
            files.extend(files_shuffle[:n_shuffle])

        if n_parts_seq > 0:
            files_parts_seq = glob(pool_base + '/parts-seq/labels/*.pgn')
            print('files_parts_seq', n_parts_seq, '/', len(files_parts_seq));
            files.extend(files_parts_seq[:n_parts_seq])

        if n_parts_ec > 0:
            files_parts_ec = glob(pool_base + '/parts-ec/labels/*.pgn')
            random.shuffle(files_parts_ec)
            print('files_parts_ec', n_parts_ec, '/', len(files_parts_ec));
            files.extend(files_parts_ec[:n_parts_ec])

        if n_parts_ec_v2 > 0:
            files_parts_ec_v2 = glob(pool_base + '/parts-ec-v2/labels/*.pgn')
            random.shuffle(files_parts_ec_v2)
            print('files_parts_ec_v2', n_parts_ec_v2, '/', len(files_parts_ec_v2));
            files.extend(files_parts_ec_v2[:n_parts_ec_v2])

        N_train = int(N * .8)
        print('train, valid', N_train, N - N_train)

        # shuffle
        random.shuffle(files)
        print('total', len(files))

        if len(files) != N:
            print('total nao bate. Esperado=', N, 'obtido=', len(files))
            return

        if test:
            print('test. Done!')
            return

        # copia
        for i in range(0, len(files)):
            if i % 100 == 0:
                print(i, '....')

            f = files[i]
            folder = 'train' if i < N_train else 'valid'

            # copia
            dest_file = os.path.join(root, folder, 'labels', Path(f).name)
            shutil.copyfile(f, dest_file)
            shutil.copyfile(
                f.replace('labels', 'images').replace('pgn', 'jpg'),
                dest_file.replace('labels', 'images').replace('pgn', 'jpg'))

        # os de teste copia todos para train
        if include_all_except_test:
            files = glob(pool_base + '/torneio-except-test/labels/*.pgn')
            print('copiando ', len(files), ' para train')
            for f in files:
                folder = 'train'
                dest_file = os.path.join(root, folder, 'labels', Path(f).name)
                shutil.copyfile(f, dest_file)
                shutil.copyfile(
                    f.replace('labels', 'images').replace('pgn', 'jpg'),
                    dest_file.replace('labels', 'images').replace('pgn', 'jpg'))

    @staticmethod
    def generate():
        DatasetGenerator.generate_dataset_v2_2lines(ver='dataset-2lines-v.2.0.1',
                                                    n_hand=100,
                                                    n_parts=100,
                                                    n_sync=0,
                                                    n_shuffle=0,
                                                    n_parts_seq=0,
                                                    n_parts_ec=0,
                                                    n_parts_ec_v2=1200,
                                                    include_all_except_test=False,
                                                    test=False)
