import os
import uuid
from glob import glob
import random

import numpy as np

import config
from game_data import GameData
from img_utils import resize
from io_utils import write_image, write_label, load_image_


class TemplateGenerator:

    def generate(self):
        part = load_image_(os.path.join(config.ROOT_DIR, 'templates/part.jpg'));
        part = resize(part, 320, 80)

        line = np.concatenate([part] * 4, axis=1)
        page = np.concatenate([line] * 25, axis=0)

        write_image(os.path.join(config.ROOT_DIR, 'templates/part-page.jpg'), page)


if __name__ == "__main__":
    TemplateGenerator().generate()
