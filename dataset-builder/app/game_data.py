import re
import random

import config
from vocab import load_vocab_175


def filter_chaves(s):
    _in = False
    r = []
    for x in s:
        if _in:
            if x == '}':
                _in = False
            continue

        # not in
        if x == '{':
            _in = True
            continue

        # not in & x!= '{'
        r.append(x)
    return ''.join(r)


def filter_numeros(line):
    return re.sub(r"[W,B]\d+\.", '', line)


def filter_pt(line):
    return line.replace('R', 'T').replace('K', 'R').replace('Q', 'D').replace('N', 'C')


def _filter(lines):
    lines = [re.split(r'###', line)[1].lstrip().rstrip() for line in lines]
    lines = [filter_chaves(line) for line in lines]
    lines = [filter_numeros(line) for line in lines]
    # lines_pt = [filter_pt(line) for line in lines]
    return lines


class GameData:

    @staticmethod
    def load(qtd, shuffled=False,
             f=config.ROOT_DIR + '/pgn/35million/all_with_filtered_anotations_since1998--250K-pt2.txt',
             inwords=None,
             seq_length=16):
        file = open(f)
        lines = file.readlines()[6:]
        lines = _filter(lines)
        jogos = [ln.split() for ln in lines]

        if shuffled:
            random.shuffle(jogos)

        if inwords:
            inwords_set = set(inwords)
        result = []
        for jogo in jogos:
            if len(jogo) < seq_length:
                continue

            if inwords is not None:
                # all in words?
                if not set(jogo[0:seq_length]).issubset(inwords_set):
                    continue

            result.append(jogo[:seq_length])
            if len(result) >= qtd:
                break

        if len(result) != qtd:
            raise Exception("GameData.load. Pedido {}, obtido: {}".format(qtd, len(result)))
        return result


if __name__ == "__main__":
    r = GameData.load(100000, shuffled=True, inwords=load_vocab_175())
    print('obtidos: ', len(r))
