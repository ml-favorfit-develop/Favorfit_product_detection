import argparse
import os
from datetime import datetime
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import ProductDetectionDataset
from transformer import ImgAugTransformer
from models import Classifier

from torch.utils.data import random_split, DataLoader
from dotenv import load_dotenv
import logger, ckpt_saver

device = "cpu"


def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def train(epoch, model, criterion, optimizer, train_loader):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    for idx, (inputs, labels) in tqdm(enumerate(train_loader), leave=False, desc=f"Epcoh:{epoch} train", total=len(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels)

        running_loss += loss.item()
        running_acc += acc.item()

    train_loss = running_loss / len(train_loader.dataset) * 100
    train_acc = running_acc / len(train_loader.dataset) * 100
    current_lr = optimizer.param_groups[0]['lr']

    logger.wnb_write({"train_Loss": train_loss, "train_Acc": train_acc, "lr": current_lr})
    print(f'Epoch:{epoch} Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')


def val(epoch, model, criterion, val_loader):
    model.eval()

    conf_labels = []
    conf_preds = []
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0

        for idx, (inputs, labels) in tqdm(enumerate(val_loader), leave=False, desc=f"Epcoh:{epoch} val", total=len(val_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            acc = torch.sum(preds == labels)

            running_loss += loss.item()
            running_acc += acc.item()

            conf_labels.extend(labels.cpu().numpy())
            conf_preds.extend(preds.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset) * 100
        val_acc = running_acc / len(val_loader.dataset) * 100

    if epoch % 10 == 0:
        logger.wnb_write_conf_mat(epoch, conf_labels, conf_preds, args.numclasses)
    logger.wnb_write({"val_Loss": val_loss, "val_Acc": val_acc})
    print(f'Epoch:{epoch} Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    model = Classifier(base_model=args.model, num_classes=args.numclasses)

    if args.load_state is not None:
        model.load_state_dict(torch.load(args.load_state))

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1, step_size_up=1000, step_size_down=1000, cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    
    logger.wnb_page_init(epochs=args.epochs, project=args.project_name, name=args.run_name, wnb_api_key=os.getenv("WANDB_KEY"))
    logger.wnb_watch(model, criterion)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    dataset = ProductDetectionDataset( \
        DB_HOST=os.getenv("DB_HOST"),
        DB_NAME=os.getenv("DB_NAME"),
        COLLECTION_NAME=os.getenv("COLLECTION_NAME"),
        transformer=ImgAugTransformer(height=args.height, width=args.width),
        label_map=args.label_map,
        height_and_width=(args.height, args.width)
    )

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader)
        val(epoch, model, criterion, val_loader)

        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        scheduler.step()

        if epoch % 10 == 0:
            ckpt_saver.save_model_pth(model_state_dict, epoch)

    logger.wnb_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=50, help='torch random seed')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--project_name', type=str, default="my_project", help='wandb log start project name')
    parser.add_argument('--run_name', type=str, default=f"{datetime.today().month}.{datetime.today().day}.start", help='wandb log start run name')
    parser.add_argument('--model', type=str, default="efficientnet_b4", help='model name')
    parser.add_argument('--numclasses', type=int, default=2, help='num of classes')
    parser.add_argument('--load_state', type=str, default=None, help='model_state.pth file path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--height', type=int, default=380, help='resize image height')
    parser.add_argument('--width', type=int, default=380, help='resize image width')
    parser.add_argument('--label_map', type=dict, default={"non_product": 0, "with_model": 0, "product": 1}, help='labeling map on DB collection')
    parser.add_argument('--batch_size', type=int, default=32, help='barch size in dataloader depend on GPU')
    parser.add_argument('--workers', type=int, default=4, help='cpu core num')
    args = parser.parse_args()

    load_dotenv()
    set_device()
    run(args)
