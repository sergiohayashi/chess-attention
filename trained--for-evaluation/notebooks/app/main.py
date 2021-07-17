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


if __name__ == '__main__':
    print('PyCharm')
    # predict()
    # train('testFly')
    # predictFrom('testFly')
    train_curriculum()
    print('Done!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
