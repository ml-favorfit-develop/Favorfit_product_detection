from model import remove_background
import dataset, logger
import os
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm

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

    with torch.no_grad():
        for batch, (img_path, tag, file_name) in tqdm(enumerate(dataloader), leave=False, desc=f"{file_name}: ", total=len(dataloader)):
            if img_path == None:
                logger.save_log(f"{batch} , {img_path} , NONE")
                continue
            
            if batch <= skip_batch:
                continue
            
            try:
                if torch.cuda.is_available():
                    img_path = img_path.to(device)
                
                output = model(img_path)

                logger.copy_and_save_image(image_array=output, image_path=img_path, tag=tag, file_name=file_name+"_mask")
                logger.save_log(f"{batch} , {img_path} , SUCCESS")
            except:
                logger.save_log(f"{batch} , {img_path} , FAILED")


def run():
    process_dataset = dataset.RemoveBackgroundDataset(\
                        DB_HOST=os.environ("DB_HOST"),
                        DB_NAME=os.environ("DB_NAME"),
                        COLLECTION_NAME=os.environ("COLLECTION_NAME"),
                        target_label=None)
    
    process_loader = DataLoader(process_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)  # inspyrenet의 경우 batch_size = 1

    
    model = remove_background.BackGroundRemover(base_model="inspyrenet", 
                                                fast=True, jit=True, device=device)

    process(model, process_loader, skip_batch=-1)


if __name__=="__main__":
    load_dotenv(".env")
    set_device()
    run()