import torch.nn as nn



class MultiModalLoss(nn.Module):
    def __init__(self):
        super(MultiModalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, modality_gt1, modality_gt2, modality_predict1, modality_predict2, com_img_w1, com_img_w2):
        losses = {}
        self_supervised_modality1 =  self.l1_loss(modality_gt1, modality_predict1)
        losses['self_supervised_modality1'] = self_supervised_modality1
        self_supervised_modality2 = self.l1_loss(modality_gt2, modality_predict2)
        losses['self_supervised_modality2'] = self_supervised_modality2
        self_supervised_common =  self.l1_loss(com_img_w1, com_img_w2)
        losses['self_supervised_common'] = self_supervised_common
        losses['total_loss'] = self_supervised_modality1 + self_supervised_modality2 + self_supervised_common
        return losses
