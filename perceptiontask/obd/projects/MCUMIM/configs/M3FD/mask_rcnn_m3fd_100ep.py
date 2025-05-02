from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.m3fd_loader_lsj import dataloader,image_size


model = model_zoo.get_config("common/models/mask_rcnn_mcumim.py").model


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = False
train.ddp.fp16_compression = False
train.ddp.find_unused_parameters=True
# train.init_checkpoint = (
#     "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
# )
train.init_checkpoint = ('/home/lpw/fastssd/lpw/DeFusionv2/experiments/MultiModalFastMV5_mfm/models/detectmodelv1.pth')

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# model.backbone.square_pad = image_size # may not work
model.roi_heads.mask_in_features=None
model.roi_heads.num_classes = 6
model.roi_heads.box_predictor.test_score_thresh = 0.02
model.roi_heads.box_predictor.test_topk_per_image = 300
model.roi_heads.box_predictor.use_sigmoid_ce = True
model.roi_heads.box_predictor.use_fed_loss = True
model.roi_heads.box_predictor.get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)

# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
# 100 ep = 84000 * 4 / 3360

train.max_iter = 84000
train.eval_period = 3000

lr_multiplier.scheduler.milestones = [69440, 75230]

# train.max_iter = 168000
# train.eval_period = 6000
#
# lr_multiplier.scheduler.milestones = [140880, 150460]
# lr_multiplier.scheduler.num_updates = train.max_iter
# lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
