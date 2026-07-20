class ImageChunk:
    def __init__(self, data, format):
        self.data = data
        self.format = format


class EncodedTextChunk:
    def __init__(self, text):
        self.text = text


class ModelInput:
    def __init__(self, chunks):
        self.chunks = chunks


class TinkerClient:
    def sample(self, model_path, model_input):
        raise RuntimeError("Synthetic smoke-test stub forces classify_image() to fail.")
