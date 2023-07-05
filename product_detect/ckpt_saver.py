import json
import torch

def save_dictionary(json_like):
    with open("./ckpt/results.json", mode="w") as f:
        json.dump(json_like, f, ensure_ascii=False, indent=4)
    return True

def save_model_pth(model_state_dict, epoch):
    torch.save(model_state_dict, f"./ckpt/model_{epoch}.pth")
    return True
