from pymongo import MongoClient
from pymongo import MongoClient, ASCENDING

import os

import shutil
import os
from tqdm import tqdm

import multiprocessing as mp
from joblib import Parallel, delayed

destination_base_path = r"media/mlfavorfit/One Touch/FAVORFIT/mask_true"

def make_path(document):
        path = os.path.join("/",
                                document["start_path"],
                                document["HDD_name"],
                                "mask",
                                document["tag"][0],
                                document["file_name"])
        return path

def make_destination_folder_path(document):
        path = os.path.join("/",
                                destination_base_path,
                                document["tag"][0])
        return path
def make_destination_path(document):
        path = os.path.join("/",
                                destination_base_path,
                                document["tag"][0],
                                document["file_name"])
        return path

def copy_files(file, destination_folder, detination_file_path):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    shutil.copy2(file, detination_file_path)

def run(file, destination_folder_path, destination_file_path):
    try:
        copy_files(file, destination_folder_path, destination_file_path)
        copy_files(file+"_mask", destination_folder_path, destination_file_path+"_mask")
    except:
        print("error")

if __name__ == '__main__':


    client = MongoClient()
    db = client[""]
    collection = db[""]
    cursor = collection.find({"label": {"$eq": "product"}})

    datas = [cur for cur in cursor]
    path_datas = [(make_path(cur),make_destination_folder_path(cur),make_destination_path(cur)) for cur in datas]

    result_arr = Parallel(n_jobs=mp.cpu_count() - 1)(delayed(run)(*paths) for paths in path_datas)

    print(sum(result_arr))