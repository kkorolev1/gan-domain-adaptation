from degan.base.base_metric import BaseMetric
class DummyMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    def __call__(self, **kwargs):
        raise NotImplementedError()

__all__ = [
    "DummyMetric"
]