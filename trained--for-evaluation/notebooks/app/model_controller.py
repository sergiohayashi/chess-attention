from evaluator import Evaluator
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
        self.model = None

    def load(self):
        self.model = AttentionEncoderDecoderModel().build()

    def useModel(self, model):
        self.model = model
        return self

    def restoreFromBestCheckpoint(self):
        bestCheckpointPath = '../best_checkpoint/1006/checkpoints/train'
        self.model.steps.restoreFromLatestCheckpoint(bestCheckpointPath)

    def restoreFromCheckpointName(self, trainName):
        checkPointPath = '../train-folder/checkpoints/' + trainName
        self.model.steps.restoreFromLatestCheckpoint(checkPointPath)

    def predictOneImage(self, imagePath):
        result = self.model.steps.evaluate(imagePath)
        return result

    def evaluateForTest(self):
        evaluator = Evaluator(self.model)
        evaluator.evaluate_test_data()

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
    if os.path.isdir(uncompressFolder):
        print(uncompressFolder, ' already exists. Skip uncompress..')
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
        self.trainer = None

    def load(self):
        self.model = AttentionEncoderDecoderModel().build()

    def useModel(self, model):
        self.model = model

    def initTrainSession(self):
        self.trainer = TrainerController(self.model)

    def prepareDatasetForTrain(self, datasetZipFile):
        # uncompress for train
        print('preparing dataset from zip file ', datasetZipFile)
        uncompressFolder = '../train-folder/tmp/' + os.path.basename(datasetZipFile).replace('.zip', '')
        uncompressToFolder(datasetZipFile, uncompressFolder)

        # prepare dataset
        self.trainer.prepareFilesForTrain(uncompressFolder)
        print('Dataset from zip file ', datasetZipFile, ' ready for training')

    def trainUntil(self, target_loss, max_epoch):
        print('starting trainUntil ', target_loss, max_epoch)
        self.trainer.trainUntil(target_loss, max_epoch)
        print('starting trainUntil ', target_loss, max_epoch, ' DONE!')

    def save(self, trainName):
        checkPointPath = '../train-folder/checkpoints/' + trainName
        self.model.steps.saveCheckpointTo(checkPointPath)
        print('model saved to ' + checkPointPath)

    def trainOrContinueForCurriculum(self, curriculumName, levelsDatasetZipFiles,
                                     target_loss, max_epoch):
        for levelZipFile in levelsDatasetZipFiles:
            checkpointName = "{}--{}".format(curriculumName, os.path.basename(levelZipFile).replace('.zip', ''))

            if self.levelCheckpointExists(checkpointName):
                # se treino ja foi finalizado, somente recupera..
                ModelPredictController().useModel(self.model).restoreFromCheckpointName(checkpointName)
            else:

                # caso contrario, faz o treinamento
                self.prepareDatasetForTrain(levelZipFile)
                self.trainUntil(target_loss, max_epoch)
                self.save()

        self.save(curriculumName)
        print('treinamento de curriculo finalizado com sucesso!')
