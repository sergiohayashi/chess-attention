import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from termcolor import colored

import config


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
            if config.USE_BIG_PLOT:
                ax = fig.add_subplot(5, 4, l + 1)
            else:
                ax = fig.add_subplot(9, 8, l + 1)
            if expected is None or l >= len(expected):
                ax.set_title(result[l], fontsize=50)
            else:
                ax.set_title(result[l] + " (" + expected[l] + ")", fontsize=50)
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    def plot_attention_EACH(self, image, result, attention_plot, expected=None):
        print(image)
        temp_image = np.array(Image.open(image))

        len_result = len(result)
        for l in range(len_result):
            fig = plt.figure()
            temp_att = np.resize(attention_plot[l], self.model.ATTENTION_SHAPE)
            img = plt.imshow(temp_image)
            plt.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_attention_unified(image, result, attention_plot, expected=None):
        temp_image = np.array(Image.open(image))
        temp_att = np.resize(np.sum(attention_plot, axis=0), config.size_mode['attention_shape'])
        img = plt.imshow(temp_image)
        plt.imshow(temp_att, cmap='bone', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()
        if expected is not None:
            Plotter.print_result(expected, result)

    @staticmethod
    def print_result(expected, result):
        print(colored('Expected:  ', attrs=['bold']), end=' ')
        for i in range(0, len(expected)):
            print(expected[i], end=' ')
        print()
        print(colored('Predicted: ', attrs=['bold']), end=' ')
        for i in range(0, len(expected)):
            if expected[i] != result[i]:
                for c in result[i]:
                    if c not in expected[i]:
                        print(colored(c, 'red', attrs=['bold']), end='')
                    else:
                        print(colored(c, 'red'), end='')
                print(' ', end='')
            else:
                print(result[i], end=' ')
        print()
