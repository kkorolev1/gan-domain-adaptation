class BaseMetric:
    def __init__(self, name=None, epoch_level=False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.epoch_level = epoch_level
    def __call__(self, **batch):
        raise NotImplementedError()