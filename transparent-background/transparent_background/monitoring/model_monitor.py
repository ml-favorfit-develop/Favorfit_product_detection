from thop import profile, clever_format
import contextlib
import os
from torchsummary import summary
import torch

def print_and_get_flops_parameter(model, input_x):         

    if  isinstance(model, torch.jit.ScriptModule):
        flops = None
        params = sum(p.numel() for p in model.parameters())
    else:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                flops, params = clever_format(profile(model, inputs=(input_x,)))

    print(f"Flops: {flops}\nParameters: {params}")

    return {"flops":flops, "params":params}
