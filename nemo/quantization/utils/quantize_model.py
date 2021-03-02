from nemo.quantization.utils.quant_modules import *

list_all = [QuantAct, QuantLinear, QuantConv1d]

def set_percentile(model, percentile: float):
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

def adjust_range(model, scale: float):
    if type(model) in list_all:
        if type(model) == QuantAct:
            model.adjust_range(scale)
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            adjust_range(m, scale)
    elif type(model) == nn.ModuleList:
        for n in model:
            adjust_range(n, scale)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                adjust_range(mod, scale)

def set_dynamic(model, update: bool):
    if type(model) in list_all:
        if type(model) == QuantAct:
            model.dynamic = update
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            set_dynamic(m, update)
    elif type(model) == nn.ModuleList:
        for n in model:
            set_dynamic(n, update)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                set_dynamic(mod, update)

def set_update_bn(model, update: bool):
    if type(model) in list_all:
        if type(model) == QuantConv1d:
            model.set_update_bn(update)
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            set_update_bn(m, update)
    elif type(model) == nn.ModuleList:
        for n in model:
            set_update_bn(n, update)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                set_update_bn(mod, update)

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

def fix_act(model):
    freeze_model(model, [QuantAct])
