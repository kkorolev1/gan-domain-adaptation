from degan.base import BaseDataset

class AAHQDataset(BaseDataset):
    def __init__(self, root_path, size=1024):
        super().__init__(root_path, size)
    
# dataset = AAHQDataset("datasets/aahq-dataset/aligned")
# print(len(dataset))