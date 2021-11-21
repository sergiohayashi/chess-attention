class Predictor:

    def __init__(self):
        pass

    def predict(self, model, image):
        return model.steps.evaluate( image)

