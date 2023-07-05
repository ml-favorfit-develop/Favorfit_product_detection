import os
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from PIL import Image

class ProductDetectionDataset(Dataset):

    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME, transformer, label_map=None):
        client = MongoClient(DB_HOST)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        cursor = collection.find({"label":{"$eq":None}})

        self.images = [(self.make_path(doc), label_map[doc["label"]]) for doc in cursor]
        self.label_map = label_map
        self.transformer = transformer
        client.close()

        print(f"Dataset -> host={DB_HOST}, label_map:{label_map}, num_of_data:{len(self.images)}")

    def make_path(self, document):
        path = os.path.join("/",
                        document["start_path"],
                        document["HDD_name"],
                        document["middle_path"],
                        document["tag"][0],
                        document["file_name"])
        return path
        
    def __getitem__(self, idx):
        if idx >= len(self.images):
            idx = idx - len(self.images)
            transformer = self.transformer
        else:
            transformer = None

        image_path, label = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if transformer != None: image = transformer(image)
            return image, label
        except:
            # 알 수 없는 오류가 발생했을 때, dummy값으로 black 이미지와 0을 레이블로 반환
            dummy_image = torch.zeros([224, 224, 3])
            return dummy_image, 0
        
    def __len__(self):
        return len(self.images) * 2
