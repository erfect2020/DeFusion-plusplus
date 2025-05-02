
import torch
import torch.nn as nn
from semseg.models.cumaelayers.tinymim import tinymim_vit_tiny_patch16
from semseg.models.cumaelayers.multiAtten import TransformerAttenBlock as AttenBlock
from torchvision.transforms.functional import normalize
from semseg.models.cumaelayers.vit import Block
from semseg.models.cumaelayers.pos_embed import interpolate_pos_embed, interpolate_pos_encoding
from semseg.models.heads.upernet import UPerHead
# from semseg.models.heads.mmupernet import UPerHead
from semseg.models.heads.segformer import SegFormerHead

class UCMIMNetV2(nn.Module):
    def __init__(self, backbone: str = 'CMNeXt-B0', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar'], fpn1_norm='BN', out_indices=[3, 5, 7, 11]):
        super(UCMIMNetV2, self).__init__()
        self.encoder = tinymim_vit_tiny_patch16()
        # self.init_pretrained()

        decoder_embed_dim = 192
        decoder_img_dim = 768
        decoder_seg_dim = 768 * 4
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
            AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_common_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decode_common_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.decoder_unique_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_unique_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_unique_residual = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        self.decoder_unique_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.decoder_unique_residual_skipconn = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)

        self.decoder_fuse_blocks = nn.ModuleList([
            AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim)),
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.decoder_fuse_blocks.append(nn.Linear(decoder_embed_dim, decoder_img_dim))
        self.decoder_fuse_skipconn = AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim))

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        embed_dim = 192
        self.out_indices = out_indices

        self.fpn1 = nn.Sequential(
            *[nn.ConvTranspose2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim * 2) if fpn1_norm == 'SyncBN' else nn.BatchNorm2d(embed_dim * 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim * 2, embed_dim * 4, kernel_size=2, stride=2),]
        )
        self.fuse_fpn1 = AttenBlock(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.fuse_post_fpn1 = nn.Sequential(norm_layer(decoder_embed_dim),
                                            nn.Linear(decoder_embed_dim, decoder_seg_dim))

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim * 4, kernel_size=2, stride=2),
        )
        self.fuse_fpn2 = AttenBlock(decoder_embed_dim, decoder_num_heads,
                                    dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.fuse_post_fpn2 = nn.Sequential(norm_layer(decoder_embed_dim),
                                            nn.Linear(decoder_embed_dim, decoder_seg_dim))

        self.fpn3 = nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1)
        self.fuse_fpn3 = AttenBlock(decoder_embed_dim, decoder_num_heads,
                                    dim_feedforward=int(mlp_ratio * decoder_embed_dim))
        self.fuse_post_fpn3 = nn.Sequential(norm_layer(decoder_embed_dim),
                                            nn.Linear(decoder_embed_dim, decoder_seg_dim))

        self.fpn4 = nn.Sequential(nn.Conv2d(80, embed_dim * 4, kernel_size=3, padding=1, stride=2))

        self.conv1x1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1280), nn.GELU())

        self.decode_head = UPerHead(in_channels = [embed_dim * 4]*4, channel=128, num_classes = 9, scales=(1, 2, 3, 6))
        # self.decode_head = UPerHead(in_channels=[192 *4 , 192 * 4, 192 * 4, 192 * 4], in_index=[0, 1, 2, 3], channels=192, out_channels=150, num_classes=150)
        # self.decode_head = SegFormerHead(dims=[192 *4 , 192 * 4, 192 * 4, 192 * 4], embed_dim=192, num_classes=150)

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

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            msg = self.load_state_dict(checkpoint, strict=False)
            print(msg)
        pos_embed = interpolate_pos_encoding(900, 192, self.encoder, 16, (16, 16), 480, 480)
        self.encoder.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self.encoder.patch_embed.img_size = (480, 480)

    def forward(self, imgs):

        img1, img2 = imgs

        img1 = normalize(img1, self.normalize_mean, self.normalize_std)
        img2 = normalize(img2, self.normalize_mean, self.normalize_std)

        enc_feas1 = self.encoder(img1)
        enc_feas2 = self.encoder(img2)

        residual_fea1 = self.enc_norm1(enc_feas1[0]) + self.enc_norm2(enc_feas1[1])
        residual_fea2 = self.enc_norm1(enc_feas2[0]) + self.enc_norm2(enc_feas2[1])

        enc_fea1 = enc_feas1[-1]
        enc_fea2 = enc_feas2[-1]

        com_img = self.decoder_common_blocks[0](enc_fea1, enc_fea2)
        residual_com_img = self.decode_common_skipconn(residual_fea2,residual_fea1) + self.decode_common_skipconn(residual_fea1,residual_fea2)
        com_img = com_img + residual_com_img
        uni_img2 = self.decoder_unique_blocks[0](enc_fea1, enc_fea2) + self.decoder_unique_residual(enc_fea2)
        residual_uni_img2 = self.decoder_unique_skipconn(residual_fea1, residual_fea2) + self.decoder_unique_residual_skipconn(residual_fea2)
        uni_img2 = uni_img2 + residual_uni_img2
        uni_img1 = self.decoder_unique_blocks[0](enc_fea2, enc_fea1) + self.decoder_unique_residual(enc_fea1)
        residual_uni_img1 = self.decoder_unique_skipconn(residual_fea2,
                                                         residual_fea1) + self.decoder_unique_residual_skipconn(residual_fea1)
        uni_img1 = uni_img1 + residual_uni_img1

        fuse_img = self.decoder_fuse_blocks[0](com_img, uni_img1) + self.decoder_fuse_blocks[0](com_img, uni_img2)

        for blk in self.recon_blocks_mim_encoder:
            fuse_img = blk(fuse_img)

        fuse_fea1 = self.fuse_fpn1(enc_feas1[self.out_indices[0]], enc_feas2[self.out_indices[0]])
        fuse_fea1 = self.fuse_post_fpn1(fuse_fea1)
        fuse_fea1 = self.unpatchifyc(fuse_fea1[:, 1:, :], c=192, p=4)
        fuse_fea1 = self.fpn1(fuse_fea1)

        fuse_fea2 = self.fuse_fpn2(enc_feas1[self.out_indices[1]], enc_feas2[self.out_indices[1]])
        fuse_fea2 = self.fuse_post_fpn2(fuse_fea2)
        fuse_fea2 = self.unpatchifyc(fuse_fea2[:, 1:, :], c=192, p=4)
        fuse_fea2 = self.fpn2(fuse_fea2)

        fuse_fea3 = self.fuse_fpn3(enc_feas1[self.out_indices[2]], enc_feas2[self.out_indices[2]])
        fuse_fea3 = self.fuse_post_fpn3(fuse_fea3)
        fuse_fea3 = self.unpatchifyc(fuse_fea3[:, 1:, :], c=192, p=4)
        fuse_fea3 = self.fpn3(fuse_fea3)

        fuse_fea4 = self.conv1x1(fuse_img)
        fuse_fea4 = self.unpatchifyc(fuse_fea4[:, 1:, :], c=80, p=4)
        fuse_fea4 = self.fpn4(fuse_fea4)

        # print("fuse fea shapes", fuse_fea1.shape, fuse_fea2.shape, fuse_fea3.shape, fuse_fea4.shape)
        features = [fuse_fea1, fuse_fea2, fuse_fea3, fuse_fea4]

        class_feature = self.decode_head(features)
        # print("class feature", class_feature.shape)
        return class_feature
