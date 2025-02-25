from kornia.losses import js_div_loss_2d, SSIMLoss
import torch.nn as nn


class PretrainLoss(nn.Module):
    def __init__(self):
        super(PretrainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.l1_loss = None
        # self.l1_loss = SSIMLoss(window_size=11, reduction='mean')
        self.js_loss = js_div_loss_2d
        self.kl_loss = nn.KLDivLoss(log_target=True)
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, reduction='mean')

    def inital_losses(self, lossses):
        lossses['self_supervised_recon'] = 0

    def forward(self, img1, img2, gt_img, rec_img1, rec_img2):
        losses = {}
        losses['total_loss'] = 0

        self.inital_losses(losses)
        for index in range(img1.size(0)):

            rec_img1_i = rec_img1[index].unsqueeze(0)
            rec_img2_i = rec_img2[index].unsqueeze(0)

            gt_i = gt_img[index].unsqueeze(0)

            gt_img1_i = gt_i[:, 2:5, :, :]
            gt_img2_i = gt_i[:, 5:8, :, :]
            self_supervised_recon = self.l1_loss(rec_img1_i, gt_img1_i) + self.l1_loss(rec_img2_i, gt_img2_i)
            losses['self_supervised_recon'] +=  self_supervised_recon

            losses['total_loss'] += self_supervised_recon #\

        return losses
