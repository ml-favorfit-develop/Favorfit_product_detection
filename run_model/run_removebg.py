from model import remove_background
import dataset, logger
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--skipindex', type=int, default=None, help='skip image index')
parser.add_argument('--label', type=str, default=None, help='target label')
args = parser.parse_args()

device = "cpu"

def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def process(model, dataloader):
    
    model.eval()
    if args.skipindex == None:
        skipindex = -1
    else:
        skipindex = args.skipindex

    with torch.no_grad():

        pbar = tqdm(total=len(dataloader), leave=False)
        for batch, (image, img_path, tag, file_name, ext) in enumerate(dataloader):

            pbar.set_description(f"{batch+skipindex+1} {file_name}: ")

            if torch.cuda.is_available():
                try:
                    image = image.to(device)
                except:
                    pass
            
            if img_path == None:
                logger.save_log(f"{batch+skipindex+1},{img_path},{args.label},NONE")
            else:
                try:
                    output = model(image)
                    
                    logger.copy_and_save_image(image_array=output, image_path=img_path, tag=tag, file_name=file_name + "_mask", ext=ext)
                    logger.save_log(f"{batch+skipindex+1},{img_path},{args.label},SUCCESS")
                except:
                    logger.save_log(f"{batch+skipindex+1},{img_path},{args.label},FAILED")
            
            pbar.update()
            
    pbar.close()


def run():
    process_dataset = dataset.RemoveBackgroundDataset(\
                        DB_HOST=os.getenv("DB_HOST"),
                        DB_NAME=os.getenv("DB_NAME"),
                        COLLECTION_NAME=os.getenv("COLLECTION_NAME"),
                        skip_index=args.skipindex,
                        target_label=args.label)
    
    # process_loader = DataLoader(process_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    process_loader = process_dataset    # inspyrenet의 경우

    model = remove_background.BackGroundRemover(base_model="inspyrenet", 
                                                fast=True, jit=False, device=device)

    process(model, process_loader)


if __name__=="__main__":
    load_dotenv()
    set_device()
    
    run()
