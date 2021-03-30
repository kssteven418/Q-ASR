from nemo.quantization.utils.quant_modules import *

list_all = [QuantAct, QuantLinear, QuantConv1d]

def set_percentile(model, percentile: float):
    """
    Recursively set the percentile of QuantAct ops.
    """
    if type(model) in list_all:
        if type(model) == QuantAct:
            model.set_percentile(percentile)
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            set_percentile(m, percentile)
    elif type(model) == nn.ModuleList:
        for n in model:
            set_percentile(n, percentile)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                set_percentile(mod, percentile)

def set_dynamic(model, dynamic: bool):
    """
    Recursively set the dynamic quantization mode of QuantAct ops.
    """
    if type(model) in list_all:
        if type(model) == QuantAct:
            model.dynamic = dynamic
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            set_dynamic(m, dynamic)
    elif type(model) == nn.ModuleList:
        for n in model:
            set_dynamic(n, dynamic)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                set_dynamic(mod, dynamic)

def freeze_model(model, freeze_list):
    """
    Freezes operations in `freeze_list` by a resursively invocation of ops.fix() for 
    the ops in `freeze_list` and ops.unfix() for the remaining ops.
    """
    if type(model) in list_all:
        if type(model) in freeze_list:
            model.fix()
        else:
            model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m, freeze_list)
    elif type(model) == nn.ModuleList:
        for n in model:
            freeze_model(n, freeze_list)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                freeze_model(mod, freeze_list)

def evaluate(model):
    "Evaluation mode - fix all operations"
    freeze_model(model, list_all)

def train(model):
    "Train mode - unfix all operations"
    freeze_model(model, [])

def calibrate(model):
    "Calibration mode - only unfix QuantAct"
    freeze_model(model, [QuantConv1d, QuantLinear])
