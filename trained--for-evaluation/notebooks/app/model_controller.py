from models import AttentionEncoderDecoderModel
import zipfile
import os
from predictor import Predictor
from trainer import TrainerController

'''
  model = ModelController();
  model.load()

  // para test
  model.initTrainSession()
  model.train( 'dataset-x')
  model.save()

  // se tiver começado, continua onde parou, caso contrario inicia e salva
  model.trainOrContinueForCurriculum( 'curriculum-1', 200, 0.1, [ 'level1', 'level2', 'level3'])

    // salva nivel corrente em 'current-level.txt'
    // salva log de acuracia e loss para cada step, com informacoes de data,hora,level,epoch
    // cria checkpoint manager e salva 


  model.restoreFromCurriculum( 'curriculum-1')
  model.predict()
'''


class ModelPredictController:

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


def uncompressToFolder(zipFile, uncompressFolder):

    if os.path.isdir( uncompressFolder):
        print( uncompressFolder, ' already exists. Skip uncompress..')
        return

    print('unzipping ', zipFile, ' to ', uncompressFolder)
    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
        zip_ref.extractall(uncompressFolder)
    print('unzipping ', zipFile, ' to ', uncompressFolder, ' done!')


class ModelTrainController:
    def __init__(self):
        self.bestCheckpointPath = 'C:/mestrado/repos-github/chess-attention/trained--for-evaluation/notebooks' \
                                  '/best_checkpoint/1006/checkpoints/train '
        self.model = None

    def load(self):
        self.model = AttentionEncoderDecoderModel().build()

    def initTrainSession(self):
        pass

    def train(self, trainName, datasetZipFile):
        # uncompress for train
        uncompressFolder = '../train-folder/tmp/' + os.path.basename(datasetZipFile).replace('.zip', '')
        uncompressToFolder(datasetZipFile, uncompressFolder)

        # trainer session
        trainer = TrainerController(self.model)

        # prepare dataset
        trainer.prepareFilesForTrain(uncompressFolder)

        trainer.trainUntil(0.1, 10)

        '''
        self.model.steps.saveCheckpointTo('../train/checkpoints/' + trainName)
        '''

    def save():
        pass
