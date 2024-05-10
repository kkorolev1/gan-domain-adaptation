class BaseMetric:
    def __init__(self, name=None, iter_based=True):
        self.name = name if name is not None else type(self).__name__
        self.iter_based = iter_based
    def __call__(self, **batch):
        raise NotImplementedError()
