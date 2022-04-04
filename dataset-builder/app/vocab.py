from config import ROOT_DIR


def load_vocab_175():
    with open(ROOT_DIR + '/vocabulary/lances--175.pgn') as f:
        line = f.readline().strip()
    return line.split()


if __name__ == "__main__":
    vocab = load_vocab_175()
    print('len = ', len(vocab))
    print('vocab: ', sorted(vocab))
