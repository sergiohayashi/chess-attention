from model_controller import ModelController


class Service:
    def __init__(self):
        pass

    def predictOneImage(self, imgFile, showAttention):
        model = ModelController()
        model.load()
        model.restoreFromCheckpoint()
        return model.predictOneImage(imgFile)

    def predictZip(self, zipFile, showAttention, calculateCir):
        pass

    def unzipAndLoadFileNames(self, zipFile):
        pass
