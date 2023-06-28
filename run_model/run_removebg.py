from model import remove_background
import dataset, logger
import os
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--skipindex', type=int, default=-1, help='skip image index')
args = parser.parse_args()

device = "cpu"

def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def process(model, dataloader, skip_batch=-1):
    
    model.eval()
    pbar = tqdm(total=len(dataloader), leave=False)

    with torch.no_grad():
        for batch, (image, img_path, tag, file_name, ext) in enumerate(dataloader):
            pbar.set_description(f"{batch} {file_name}: ")
            if batch <= skip_batch:
                pass
            if torch.cuda.is_available():
                try:
                    image = image.to(device)
                except:
                    pass
            
            if img_path == None:
                logger.save_log(f"{batch} , {img_path} , NONE")
            else:
                try:
                    output = model(image)
                    
                    logger.copy_and_save_image(image_array=output, image_path=img_path, tag=tag, file_name=file_name + "_mask." + ext)
                    logger.save_log(f"{batch},{img_path},SUCCESS")
                except:
                    logger.save_log(f"{batch},{img_path},FAILED")
            
            pbar.update()
            
    pbar.close()


def run():
    process_dataset = dataset.RemoveBackgroundDataset(\
                        DB_HOST=os.getenv("DB_HOST"),
                        DB_NAME=os.getenv("DB_NAME"),
                        COLLECTION_NAME=os.getenv("COLLECTION_NAME"),
                        target_label=None)
    
    # process_loader = DataLoader(process_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    process_loader = process_dataset    # inspyrenet의 경우

    model = remove_background.BackGroundRemover(base_model="inspyrenet", 
                                                fast=True, jit=True, device=device)

    process(model, process_loader, skip_batch=args.skipindex)


if __name__=="__main__":
    load_dotenv()
    set_device()
    
    run()
