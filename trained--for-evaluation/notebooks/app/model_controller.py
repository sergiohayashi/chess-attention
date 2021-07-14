from models import AttentionEncoderDecoderModel

from predictor import Predictor


class ModelController:

    def __init__(self):
        self.bestCheckpointPath = 'C:/mestrado/repos-github/chess-attention/trained--for-evaluation/notebooks' \
                                  '/best_checkpoint/1006/checkpoints/train '
        self.model = None

    def load(self):
        self.model = AttentionEncoderDecoderModel().build()

    def restoreFromBestCheckpoint(self):
        self.model.steps.restoreFromLatestCheckpoint()

    def predictOneImage(self, imagePath):
        result = self.model.steps.evaluate(imagePath)
        return result

    def predictZip(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def trainWith(self, level):
        pass

    def resetMetrics(self):
        pass

    def getConfig(self):
        pass

    def getMetrics(self):
        pass
