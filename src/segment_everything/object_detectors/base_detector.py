class BaseDetector:
    def __init__(self, model_path, trainable=False):
        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1]
        self.trainable = trainable

    def train(self, training_data):
        raise NotImplementedError()

    def predict(self, image_data):
        raise NotImplementedError()
