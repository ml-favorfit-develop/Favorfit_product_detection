from PIL import Image
import numpy as np
import os
from pymongo import MongoClient
from torch.utils.data import Dataset
import torch

Image.MAX_IMAGE_PIXELS = None


class ProductDetectionDataset(Dataset):
    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME, transformer, label_map=None, height_and_width=(224, 224)):
        client = MongoClient(DB_HOST)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        cursor = collection.find({"label": {"$ne": None}})
        self.images = [(self.make_path(doc), label_map[doc["label"]]) for doc in cursor]
        self.images_aug = [(image, label) for image, label in self.images if label == 1]

        self.label_map = label_map
        self.transformer = transformer
        self.height_and_width = height_and_width
        client.close()

        print(
            f"Dataset -> host={DB_HOST}, label_map:{label_map}, num_of_data:{len(self.images) + len(self.images_aug)}")

    def make_path(self, document):
        path = os.path.join("/",
                            document["start_path"],
                            document["HDD_name"],
                            document["middle_path"],
                            document["tag"][0],
                            document["file_name"])
        return path

    def __getitem__(self, idx):
        if idx < len(self.images):
            need_aug = False
            image_path, label = self.images[idx]
        else:
            idx = idx - int(len(self.images))
            need_aug = True
            image_path, label = self.images_aug[idx]

        try:
            x = np.array(Image.open(image_path).convert('RGB'))
            image = self.transformer(x, apply_aug=need_aug)
            return image, label
        except:
            # 알 수 없는 오류가 발생했을 때, dummy값으로 black 이미지와 0을 레이블로 반환
            dummy_image = torch.zeros([3, self.height_and_width[0], self.height_and_width[1]])
            return dummy_image, 0

    def __len__(self):
        return len(self.images) + len(self.images_aug)
    