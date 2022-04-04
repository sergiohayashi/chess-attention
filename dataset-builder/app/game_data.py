import re
from random import random


class GameData:

    def load_from(self, n, shuffled=False, f='../pgn/35million/all_with_filtered_anotations_since1998--250K-pt2.txt'):
        file = open(f)
        lines = file.readlines()[6:]

        if shuffled:
            random.shuffle(lines)

        return _filter(lines)

    @staticmethod
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

    # filter_chaves( 'jdflkasjd{djklsafd} dfasdjf f{dkjfl}jk {}x)')

    @staticmethod
    def filter_numeros(line):
        return re.sub(r"[W,B]\d+\.", '', line)

    @staticmethod
    def filter_pt(line):
        return line.replace('R', 'T').replace('K', 'R').replace('Q', 'D').replace('N', 'C')

    @staticmethod
    def _filter(lines):
        lines = [re.split(r'###', line)[1].lstrip().rstrip() for line in lines]
        lines = [filter_chaves(line) for line in lines]
        lines = [filter_numeros(line) for line in lines]
        lines_pt = [filter_pt(line) for line in lines]
        return lines, lines_pt

