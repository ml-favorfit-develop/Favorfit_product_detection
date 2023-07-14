from PIL import Image
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
        self.images = [(self.make_path(doc), label_map[doc["label"]]) for doc in cursor] * 2

        self.label_map = label_map
        self.transformer = transformer
        self.height_and_width = height_and_width
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
        if idx < int(len(self.images)/2):
            need_aug = False
            image_path, label = self.images[idx]
        else:
            need_aug = True
            image_path, label = self.images[idx]

        # try:
        x = Image.open(image_path).convert('RGB')
        image = self.transformer(x, apply_aug=need_aug)
        return image, label
        # except:
        #     # 알 수 없는 오류가 발생했을 때, dummy값으로 black 이미지와 0을 레이블로 반환
        #     dummy_image = torch.zeros([3, self.height_and_width[0], self.height_and_width[1]])
        #     return dummy_image, 0

    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    from transformer import ImgAugTransformer
    from dotenv import load_dotenv

    load_dotenv()
    dataset = ProductDetectionDataset( \
        DB_HOST=os.getenv("DB_HOST"),
        DB_NAME=os.getenv("DB_NAME"),
        COLLECTION_NAME=os.getenv("COLLECTION_NAME"),
        transformer=ImgAugTransformer(height=380, width=380),
        label_map={"non_product":0, "with_model":0, "product":1},
        height_and_width=(380, 380)
    )

    
    print(1 / torch.tensor([cur[1] for cur in dataset.images]).bincount())