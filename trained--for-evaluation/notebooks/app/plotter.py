import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Plotter:
    def __init__(self, model):
        self.model = model

    def plot_attention(self, image, result, attention_plot, expected=None):
        print(image)
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(50, 50))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], self.model.ATTENTION_SHAPE)
            ax = fig.add_subplot(8, 9, l + 1)
            if expected is None or l >= len(expected):
                ax.set_title(result[l], fontsize=40)
            else:
                ax.set_title(result[l] + " (" + expected[l] + ")", fontsize=40)
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()
