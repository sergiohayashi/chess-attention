# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def predict():
    model = ModelController();
    model.load()
    # model.model.print_summary()
    model.restoreFromBestCheckpoint()
    # model.model.print_summary()
    print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def train():
    pass




if __name__ == '__main__':
    from model_controller import ModelController
    print('PyCharm')
    # predict()
    train()
    print('Done!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
