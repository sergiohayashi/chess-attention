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


def train():
    model = ModelTrainController();
    model.load()
    # model.initTrainSession()
    model.train('test-fly', "../train-folder/dataset/nivel-0--dataset-v034--2lines-parts--42k.zip")


if __name__ == '__main__':
    print('PyCharm')
    # predict()
    train()
    print('Done!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
