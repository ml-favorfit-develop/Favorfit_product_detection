from model import Classifier
from dataset import ProductDetectionDataset
from  transformer import ImgResizeTransformer
import logger
from db import DBAdmin

import os
import torch
from torch.utils.data import DataLoader

from dotenv import load_dotenv
from tqdm import tqdm
import argparse

device = "cpu"

def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def run(args):
    model = Classifier(base_model=args.model, num_classes= args.numclasses)
    db_admin = DBAdmin(DB_HOST=os.getenv("DB_HOST"), DB_NAME=os.getenv("DB_NAME"), COLLECTION_NAME=os.getenv("COLLECTION_NAME"))

    if args.loadstate is not None:
        model.load_state_dict(torch.load(args.loadstate))

    dataset = ProductDetectionDataset( \
        DB_HOST=os.getenv("DB_HOST"),
        DB_NAME=os.getenv("DB_NAME"),
        COLLECTION_NAME=os.getenv("COLLECTION_NAME"),
        transformer=ImgResizeTransformer(height=args.height, width=args.width),
        skip_index=args.skipindex,
        height_and_width=(args.height, args.width)
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batchsize, num_workers=args.workers, pin_memory=True, shuffle=False)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)


    total_data_points_processed = 0
    for idx, (inputs, paths, db_id) in tqdm(enumerate(dataloader), desc="Running: ", total=len(dataloader)):
        
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        
        for idx, label in enumerate(outputs):
            db_admin.put_field(db_id[idx], "label_model", args.labelmap[label.item()])
            logger.save_log(f"{label}, {paths[idx]}, {total_data_points_processed + idx + (0 if args.skipindex == None else args.skipindex + 1)}")
        total_data_points_processed += len(inputs)



if __name__ == "__main__":
    load_dotenv(".env")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="efficientnet_b0", help='model name')
    parser.add_argument('--numclasses', type=int, default=2, help='num of classes')
    parser.add_argument('--loadstate', type=str, default="../product_detect/ckpt/latest.pth", help='model_state.pth file path')
    parser.add_argument('--workers', type=int, default=4, help='cpu core num')
    parser.add_argument('--height', type=int, default=224, help='resize image height')
    parser.add_argument('--width', type=int, default=224, help='resize image width')
    parser.add_argument('--skipindex', type=int, default=None, help='skip index to resume')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size in dataloader depend on GPU')
    parser.add_argument('--labelmap', type=dict, default={0:"non_product", 1:"product"}, help="when update DB, label saved like this option")
    args = parser.parse_args()

    run(args)
    # path = "../product_detect/ckpt/latest.pth"