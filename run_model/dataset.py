import os
from torch.utils.data import Dataset
from pymongo import MongoClient, ASCENDING
from PIL import Image

class RemoveBackgroundDataset(Dataset):
    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME, target_label=None):
        client = MongoClient(DB_HOST)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        docs = collection.find({"label":{"$ne":target_label}}.sort({"_id": ASCENDING}))

        self.images = [(self.make_path(doc), docs["tag"][0], docs["file_name"]) for doc in docs]

        client.close()

    def make_path(self, document):
        path = os.path.join("/",
                        document["start_path"],
                        document["HDD_name"],
                        document["middle_path"],
                        document["tag"][0],
                        os.path.split(document["file_name"])[0])
        return path

    def __getitem__(self, idx):
        image, tag, file_name = self.images[idx]
        try:
            image = Image.open(image).convert('RGB')
            return image, tag, file_name
        except:
            return None, "Error", "Error"
        
    def __len__(self):
        return len(self.images)
