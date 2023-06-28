import os
from torch.utils.data import Dataset
from pymongo import MongoClient, ASCENDING
from PIL import Image

class RemoveBackgroundDataset(Dataset):
    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME, target_label=None):
        client = MongoClient(DB_HOST)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        docs = collection.find({"label":{"$ne":target_label}}).sort("_id", ASCENDING)

        self.images = [(self.make_path(doc), doc["tag"][0], doc["file_name"], doc["ext"]) for doc in docs]

        client.close()

    def make_path(self, document):
        path = os.path.join("/",
                        document["start_path"],
                        document["HDD_name"],
                        document["middle_path"],
                        document["tag"][0],
                        document["file_name"])
        return path

    def __getitem__(self, idx):
        image_path, tag, file_name, ext = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            return image, image_path, tag, file_name, ext
        except:
            return None, "Error", "Error", "Error"
        
    def __len__(self):
        return len(self.images)
