from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)



# Constants
NUM_LAYERS = 15
LAYER_DECAY_RATE = 1
SEG_HEAD_SCALE = 5
FPN_SCALE = 2
FUSE_ENCODER_SCALE = 1

def get_num_layer_for_vit(var_name):
    if var_name in ("encoder.cls_token", "encoder.mask_token", "encoder.pos_embed"):
        return 0
    elif var_name.startswith("encoder.patch_embed"):
        return 0
    elif var_name.startswith("encoder.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return NUM_LAYERS - 1

def add_parameter_group(parameter_groups, group_name, param, name, scale, weight_decay, lr):
    if group_name not in parameter_groups:
        parameter_groups[group_name] = {
            "weight_decay": weight_decay,
            "params": [],
            "param_names": [],
            "lr_scale": scale,
            "group_name": group_name,
            "lr": scale * lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }
    parameter_groups[group_name]["params"].append(param)
    parameter_groups[group_name]["param_names"].append(name)

def get_layerdecay_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01):
    parameter_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if 'encoder.' in name:
            group_name = 'decay'
            layer_id = get_num_layer_for_vit(name)
            group_name = f"layer_{layer_id}_{group_name}"
            scale = LAYER_DECAY_RATE ** (NUM_LAYERS - layer_id - 1)
        elif 'decode_head.' in name or 'auxiliary_head.' in name:
            group_name = 'seg_head'
            scale = SEG_HEAD_SCALE
        elif 'fpn' in name:
            group_name = 'fpn'
            scale = FPN_SCALE
        else:
            group_name = 'fuse_encoder'
            scale = FUSE_ENCODER_SCALE

        if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
            group_name = "no_decay"
            this_weight_decay = 0.
            scale = 1
        else:
            this_weight_decay = weight_decay

        add_parameter_group(parameter_groups, group_name, param, name, scale, this_weight_decay, lr)

    return AdamW(list(parameter_groups.values()))

    # if optimizer == 'adamw':
    #     return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # else:
    #     return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)