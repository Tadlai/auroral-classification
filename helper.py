class NumpyIterable:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.as_numpy_iterator()


class TooDarkException(Exception):
    pass
