from glob import glob

from dataset_builder import DatasetGenerator
from model_controller import ModelPredictController, ModelTrainController


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def predict():
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    # model.model.print_summary()
    print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def predictFrom(trainName, n=1):
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    # model.restoreFromCheckpointName( trainName)
    model.restoreFromBestCheckpoint()
    # model.model.print_summary()
    print('------------- best checkpoint ------------------')
    model.evaluateForTest()
    model.restoreFromCheckpointName(trainName)
    print('------------- ', trainName, '------------------')
    for _ in range(0, n):
        model.evaluateForTest()
    # print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    # print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def train(trainName):
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.prepareDatasetForTrain("../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip")
    # model.trainUntil(0.01, 1)
    # model.save(trainName)


def train_curriculum():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-1', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_2():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-2', [
        #   '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        # , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_3():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-3', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        # , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        #  '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        # , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_4():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-4', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_mixed_try11():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-mixed-try11', [
        '../train-folder/dataset/mixed.zip'
    ], 0.01, 50, (25000, 500))

    '''
        Resultado:
        len 1 accuracy 0.8157894611358643 cir 0.102339186
        len 2 accuracy 0.8070175647735596 cir 0.1125731
        len 3 accuracy 0.8157894611358643 cir 0.10550683
        len 4 accuracy 0.8004385828971863 cir 0.11878655
    '''


def train_curriculum_8():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-8', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 50, (25000, 500))


def train_curriculum_10():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-10', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.01, 30, (25000, 500))


def train_curriculum_13():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-13', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.05, 0.98, 30, (25000, 500), lens=[1, 2, 3, 4])


def train_1003_5__1006_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('from-1003-5')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('_1006_', [
        '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.98, 50, (0, 0), lens=[1, 2, 3, 4])


def train_curriculum_14():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-14', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 30), (0, 0), lens=[1, 2, 3, 4])


def train_from_1003_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1003/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1005-1-', [
        '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def train_from_1005_1_():  # result: 0.86	0.81	0.83	0.79
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('local-1005-1-')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1006-1-', [
        '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def level7_try1_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('level7_try1_', [
        '../train-folder/dataset/mixed.zip'
    ], 0.025, 0.99, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def evaluate_best():
    model = ModelPredictController()
    model.load()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    model.evaluateForTest('irt_hebraica_jan2020')


def train_curriculum_21():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-21', [
        '../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../train-folder/dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../train-folder/dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../train-folder/dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        # , '../train-folder/dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 20), (1, 1), lens=[1, 2, 3, 4])


def best_add_hebraica():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('1006_add_hebraica_try2', [
        '../train-folder/tmp/mixed-com-hebraica'
    ], 0.025, 0.99, (1, 50), (1, 1), lens=[1, 2, 3, 4])


if __name__ == '__main__':
    print('PyCharm')
    # level7_try1_()

    best_add_hebraica()

    # DatasetGenerator().generate()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# TODO: Para cada novo dataset, fazer trienamento para len=1,2,3,4
# target=loss=0.1, acc=0.99

# TODO:  Por que para alguns aparece a predição UNK?
