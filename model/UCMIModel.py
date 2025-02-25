import torch.nn as nn
import torch
from models.tinymim import tinymim_vit_tiny_patch16
# from timm.models.vision_transformer import PatchEmbed, Block
from models.vit import Block
from models.multiAtten import TransformerAttenBlockVpaper as AttenBlock
import os
from torchvision.transforms.functional import normalize
from utils.pos_embed import interpolate_pos_encoding
from utils.pos_embed import get_2d_sincos_pos_embed


class UCMIMNetV3(nn.Module):
    def __init__(self):
        super(UCMIMNetV3, self).__init__()
        self.encoder = tinymim_vit_tiny_patch16()
        pretrained_ckpt = './TinyMIM-PT-Tstar.pth'
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load init checkpoint from: %s" % pretrained_ckpt)
        print("check point ", checkpoint.keys())
        checkpoint = checkpoint['model']
        self.encoder.load_state_dict(checkpoint, strict=True)
        # self.encoder.patch_embed.proj.stride = 8
        # pos_embed = interpolate_pos_encoding(729, 192, self.encoder, 16, (8, 8), 224, 224)
        # print("pos embed shape",pos_embed.shape)
        # self.encoder.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        decoder_embed_dim = 192
        decoder_img_dim = 768
        decoder_num_heads = 16
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm
        self.enc_norm1 = nn.LayerNorm(decoder_embed_dim)
        self.enc_norm2 = nn.LayerNorm(decoder_embed_dim)
        self.recon_blocks_mim_encoder = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(2)
            ])

        self.decoder_common_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=True, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_common_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decode_common_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, common=True, dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.decoder_unique_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_unique_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_unique_residual = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        self.decoder_unique_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, common=False, dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.decoder_unique_residual_skipconn = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)

        self.decoder_fuse_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, common=False, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_fuse_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_fuse_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, common=False, dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.encoder.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.mim_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.mim_decoder_norm = norm_layer(decoder_embed_dim)
        self.mim_decoder_pred = nn.Linear(decoder_embed_dim, decoder_img_dim, bias=True)

    def unpatchifyc(self, x, c=12, p=8):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked,  ids_restore, ids_keep

    def align_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_decoder(self, x, ids_restore):

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.mim_decoder_blocks:
            x = blk(x)
        x = self.mim_decoder_norm(x)

        # predictor projection
        x = self.mim_decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward(self, img1, img2):

        img1 = normalize(img1, self.normalize_mean, self.normalize_std)
        img2 = normalize(img2, self.normalize_mean, self.normalize_std)

        enc_feas1 = self.encoder(img1)
        enc_feas2 = self.encoder(img2)
        residual_fea1 = self.enc_norm1(enc_feas1[0]) + self.enc_norm2(enc_feas1[1])
        residual_fea2 = self.enc_norm1(enc_feas2[0]) + self.enc_norm2(enc_feas2[1])

        enc_fea1 = enc_feas1[-1]
        enc_fea2 = enc_feas2[-1]

        if torch.rand(1).item() > 0.5:
            com_img = self.decoder_common_blocks[0](enc_fea1, enc_fea2)
            residual_com_img = self.decode_common_skipconn(residual_fea1, residual_fea2)
        else:
            com_img = self.decoder_common_blocks[0](enc_fea2, enc_fea1)
            residual_com_img = self.decode_common_skipconn(residual_fea2, residual_fea1)

        # f torch.rand(1).item() > 0.5 else self.decoder_common_blocks[0](enc_fea2, enc_fea1)
        # residual_com_img = self.decode_common_skipconn(residual_fea1,residual_fea2) + self.decode_common_skipconn(residual_fea2,residual_fea1)

        uni_img2 = self.decoder_unique_blocks[0](enc_fea1, enc_fea2) + self.decoder_unique_residual(enc_fea2)
        residual_uni_img2 = self.decoder_unique_skipconn(residual_fea1, residual_fea2) + self.decoder_unique_residual_skipconn(residual_fea2)

        uni_img1 = self.decoder_unique_blocks[0](enc_fea2, enc_fea1) + self.decoder_unique_residual(enc_fea1)
        residual_uni_img1 = self.decoder_unique_skipconn(residual_fea2,
                                                         residual_fea1) + self.decoder_unique_residual_skipconn(residual_fea1)

        com_mimg, ids_restore, ids_keep = self.random_masking(com_img[:, 1:, :], mask_ratio=0.75)
        com_mimg = torch.cat([com_img[:, :1, :], com_mimg], dim=1)

        def align_and_concatenate(img, ids_keep):
            # Align and mask the image except for the first channel
            aligned_mimg = self.align_masking(img[:, 1:, :], ids_keep)
            # Concatenate the original image's first channel with the aligned and masked image
            mimg = torch.cat([img[:, :1, :], aligned_mimg], dim=1)
            return mimg
        uni_mimg1 = align_and_concatenate(uni_img1, ids_keep)
        uni_mimg2 = align_and_concatenate(uni_img2, ids_keep)
        residual_com_mimg = align_and_concatenate(residual_com_img, ids_keep)
        residual_uni_mimg1 = align_and_concatenate(residual_uni_img1, ids_keep)
        residual_uni_mimg2 = align_and_concatenate(residual_uni_img2, ids_keep)

        fuse_img = self.decoder_fuse_blocks[0](com_img, uni_img1) + self.decoder_fuse_blocks[0](com_img, uni_img2)
        rec_img = self.decoder_fuse_blocks[0](com_mimg, uni_mimg1) + self.decoder_fuse_blocks[0](com_mimg, uni_mimg2)
        residual_fuse_img = self.decoder_fuse_skipconn(residual_com_img, residual_uni_img1) + self.decoder_fuse_skipconn(
                 residual_com_img, residual_uni_img2)
        residual_rec_img = self.decoder_fuse_skipconn(residual_com_mimg, residual_uni_mimg1) + self.decoder_fuse_skipconn(
            residual_com_mimg, residual_uni_mimg2)

        com_img = com_img + residual_com_img
        uni_img2 = uni_img2 + residual_uni_img2
        uni_img1 = uni_img1 + residual_uni_img1

        fuse_img = fuse_img + residual_fuse_img
        fuse_main = fuse_img
        rec_img = rec_img + residual_rec_img
        #
        # fuse_img = fuse_img + residual_fuse_img

        # rec_img, ids_restore = self.random_masking(fuse_img[:, 1:, :], mask_ratio=0.75)
        # rec_img = torch.cat([fuse_img[:,:1,:], rec_img], dim=1)
        for blk in self.recon_blocks_mim_encoder:
            rec_img = blk(rec_img)
            fuse_img = blk(fuse_img)

        for blk in self.decoder_common_blocks[1:]:
            com_img = blk(com_img)
        for blk in self.decoder_unique_blocks[1:]:
            uni_img1 = blk(uni_img1)
            uni_img2 = blk(uni_img2)

        fuse_img = fuse_img + fuse_main
        # fuse_img = residual_fuse_img
        for blk in self.decoder_fuse_blocks[1:]:
            fuse_img = blk(fuse_img)

        rec_img = self.forward_decoder(rec_img, ids_restore)

        rec_img2 = self.unpatchifyc(rec_img[:, 1:, :], c=3, p=16)
        com_img = self.unpatchifyc(com_img[:, 1:, :], c=3, p=16)
        uni_img1 = self.unpatchifyc(uni_img1[:, 1:, :], c=3, p=16)
        uni_img2 = self.unpatchifyc(uni_img2[:, 1:, :], c=3, p=16)
        fuse_img = self.unpatchifyc(fuse_img[:, 1:, :], c=3, p=16)

        return rec_img2, rec_img2, com_img, uni_img1, uni_img2, fuse_img
