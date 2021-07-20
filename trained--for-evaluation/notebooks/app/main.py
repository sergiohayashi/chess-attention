from model_controller import ModelPredictController, ModelTrainController


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def predict():
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    model.restoreFromBestCheckpoint()
    # model.model.print_summary()
    print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def predictFrom(trainName):
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    # model.restoreFromCheckpointName( trainName)
    model.restoreFromBestCheckpoint()
    # model.model.print_summary()
    model.evaluateForTest()
    model.restoreFromCheckpointName(trainName)
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


if __name__ == '__main__':
    print('PyCharm')
    # predict()
    # train('testFly')
    # predictFrom('testFly')
    # train_curriculum_4()
    # train_mixed_try5()
    train_mixed_try11()
    print('Done!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#TODO: Para cada novo dataset, fazer trienamento para len=1,2,3,4
# target=loss=0.1, acc=0.99
