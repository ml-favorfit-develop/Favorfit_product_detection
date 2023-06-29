import os
from torch.utils.data import Dataset
from pymongo import MongoClient, ASCENDING
from PIL import Image

class RemoveBackgroundDataset(Dataset):
    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME, skip_index=None, target_label=None):
        client = MongoClient(DB_HOST)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        if target_label == None:
            cursor = collection.find({}).sort("_id", ASCENDING)
        else:
            cursor = collection.find({"label":{"$eq":target_label}}).sort("_id", ASCENDING)

        if skip_index != None: 
            for _ in range(skip_index+1): cursor.next()

        self.images = [(self.make_path(doc), doc["tag"][0], doc["file_name"], doc["ext"]) for doc in cursor]

        client.close()

        print(f"Dataset -> label:{target_label}, skip_index:{skip_index}, num_of_data:{len(self.images)}")

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
            return None, "Error", "Error", "Error", "Error"

    # def __getitem__(self, idx):
    #     if isinstance(idx, slice):
    #         return [self.__get_single_item(i) for i in range(*idx.indices(len(self)))]
    #     else:
    #         return self.__get_single_item(idx)

    # def __get_single_item(self, idx):
    #     image_path, tag, file_name, ext = self.images[idx]
    #     try:
    #         image = Image.open(image_path).convert('RGB')
    #         return image, image_path, tag, file_name, ext
    #     except:
    #         return None, "Error", "Error", "Error", "Error"
        
    def __len__(self):
        return len(self.images)
