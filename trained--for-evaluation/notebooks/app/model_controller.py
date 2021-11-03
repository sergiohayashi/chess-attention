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

    def __init__(self,
                 NUM_LINHAS=2,
                 NO_TEACH=True
                 ):
        self.model = None
        self.NUM_LINHAS = NUM_LINHAS
        self.NO_TEACH = NO_TEACH

    def load(self):
        self.model = AttentionEncoderDecoderModel(NUM_LINHAS=self.NUM_LINHAS, NO_TEACH=self.NO_TEACH).build()

    def useModel(self, model):
        self.model = model
        return self

    def restoreFromBestCheckpoint(self):
        bestCheckpointPath = '../best_checkpoint/1006/checkpoints/train'
        self.model.steps.restoreFromLatestCheckpoint(bestCheckpointPath)

    def restoreFromCheckpointName(self, trainName):
        checkPointPath = '../train-folder/checkpoints/' + trainName
        self.model.steps.restoreFromLatestCheckpoint(checkPointPath)

    def restoreFromCheckpointRelativePath(self, relativePath):
        self.model.steps.restoreFromLatestCheckpoint(relativePath)

    def predictOneImage(self, imagePath):
        result = self.model.steps.evaluate(imagePath)
        return result

    def evaluateForTest(self,  dataset='test', plot_attention=False, _len=4):
        evaluator = Evaluator(self.model, _len)
        evaluator.evaluate_test_data( dataset, plot_attention)

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
    def __init__(self,
                 NUM_LINHAS=2,
                 NO_TEACH=True
                 ):
        self.bestCheckpointPath = 'C:/mestrado/repos-github/chess-attention/trained--for-evaluation/notebooks' \
                                  '/best_checkpoint/1006/checkpoints/train '
        self.model = None
        self.trainer = None
        self.NO_TEACH = NO_TEACH
        self.NUM_LINHAS = NUM_LINHAS

    def load(self):
        self.model = AttentionEncoderDecoderModel(NUM_LINHAS=self.NUM_LINHAS, NO_TEACH=self.NO_TEACH).build()

    def useModel(self, model):
        self.model = model

    def initTrainSession(self, BATCH_SIZE=32):
        self.trainer = TrainerController(self.model, BATCH_SIZE=BATCH_SIZE)

    # TODO: refatorar. Reeptido de PredictController
    def restoreFromBestCheckpoint(self):
        bestCheckpointPath = '../best_checkpoint/1006/checkpoints/train'
        self.model.steps.restoreFromLatestCheckpoint(bestCheckpointPath)

    def restoreFromCheckpointName(self, trainName):
        # TODO: refatorar. Reeptido de PredictController
        checkPointPath = '../train-folder/checkpoints/' + trainName
        self.model.steps.restoreFromLatestCheckpoint(checkPointPath)

    # TODO: refatorar. Reeptido de PredictController
    def restoreFromCheckpointRelativePath(self, relativePath):
        self.model.steps.restoreFromLatestCheckpoint(relativePath)

    # TODO: refatorar. Reeptido de PredictController
    def evaluateForTest(self, dataset='test', _len=4):
        evaluator = Evaluator(self.model, _len)
        evaluator.evaluate_test_data(dataset)

    def prepareDatasetForTrain(self, datasetZipFileOrFolder, use_sample=(0.1, 0.1)):
        # uncompress for train

        if datasetZipFileOrFolder.endswith('.zip'):
            print('preparing dataset from zip file ', datasetZipFileOrFolder)
            uncompressFolder = '../train-folder/tmp/' + os.path.basename(datasetZipFileOrFolder).replace('.zip', '')
            uncompressToFolder(datasetZipFileOrFolder, uncompressFolder)
        else:
            uncompressFolder = datasetZipFileOrFolder

        # prepare dataset
        self.trainer.prepareFilesForTrain(uncompressFolder, use_sample)
        print('Dataset from zip file ', datasetZipFileOrFolder, ' ready for training')

    def trainUntil(self, target_loss, target_acc, min_max_epoch, lens=[4], train_name='none'):
        print('starting trainUntil ', target_loss, min_max_epoch, train_name)
        self.trainer.trainUntil(target_loss, target_acc, min_max_epoch, lens, train_name)
        print('starting trainUntil ', target_loss, min_max_epoch, train_name, ' DONE!')

    def save(self, trainName):
        checkPointPath = '../train-folder/checkpoints/' + trainName
        self.model.steps.saveCheckpointTo(checkPointPath)
        print('model saved to ' + checkPointPath)

    def levelCheckpointExists(self, trainName):
        checkPointPath = '../train-folder/checkpoints/' + trainName
        return self.model.steps.checkpointExists(checkPointPath)

    def trainOrContinueForCurriculum(self, curriculumName, levelsDatasetZipFiles,
                                     target_loss, target_acc, min_max_epoch, use_sample=(0.1, 0.1), lens=[4]):

        if self.levelCheckpointExists(curriculumName):
            print('treino já finalizado. Checkpoint em ', self.levelCheckpointExists(curriculumName));
            return

        skip = True
        for levelZipFile in levelsDatasetZipFiles:
            checkpointName = "{}--{}".format(curriculumName, os.path.basename(levelZipFile).replace('.zip', ''))

            if skip and self.levelCheckpointExists(checkpointName):
                # se treino ja foi finalizado, somente recupera..
                print('treino para ', checkpointName, ' ja realizado. Recupera checkpoint')
                ModelPredictController().useModel(self.model).restoreFromCheckpointName(checkpointName)
            else:
                # caso contrario, faz o treinamento
                print('treino para ', checkpointName, ' NAO realizado. Realiza treinamento.')
                self.prepareDatasetForTrain(levelZipFile, use_sample)
                self.trainUntil(target_loss, target_acc, min_max_epoch, lens, checkpointName)
                self.save(checkpointName)
                skip = False

                # valida no testset, até o tamanho maximo infoamdo
                evaluator = Evaluator(self.model, target_len=lens[-1])
                evaluator.evaluate_test_data()

        self.save(curriculumName)
        print('treinamento de curriculo finalizado com sucesso!')
