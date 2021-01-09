from nemo.quantization.utils.quant_modules import *

list_all = [QuantAct, QuantLinear, QuantConv1d]

def freeze_model(model, freeze_list):
    """
    freeze the activation range. Resursively invokes layer.fix()
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
    freeze_model(model, list_all)

def train(model):
    freeze_model(model, [])

def calibrate(model):
    freeze_model(model, [QuantConv1d, QuantLinear])
