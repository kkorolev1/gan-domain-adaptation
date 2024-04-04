from DEGAN.base import BaseDataset

class AAHQDataset(BaseDataset):
    def __init__(self, root_path):
        super().__init__(root_path)
    
# dataset = AAHQDataset("datasets/aahq-dataset/aligned")
# print(len(dataset))