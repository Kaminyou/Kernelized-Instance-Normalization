import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.util import ImagePool

from models.base import BaseModel
from models.discriminator import Discriminator
from models.generator import Generator
from models.kin import (
    init_kernelized_instance_norm,
    not_use_kernelized_instance_norm,
    use_kernelized_instance_norm,
)
from models.lsesim_loss import (
    VGG16,
    GANLoss,
    Normalization,
    PerceptualLoss,
    SpatialCorrelativeLoss,
    StyleLoss,
    cal_gradient_penalty,
)
from models.tin import (
    init_thumbnail_instance_norm,
    not_use_thumbnail_instance_norm,
    use_thumbnail_instance_norm,
)


class LSeSim(BaseModel):
    # instance normalization can be different from
    # the one specified during training
    def __init__(self, config, norm_cfg=None, isTrain=True):
        BaseModel.__init__(self, config)
        self.isTrain = isTrain
        self.attn_layers = "4, 7, 9"
        self.patch_nums = 256
        self.patch_size = 64
        self.loss_mode = "cos"
        self.use_norm = True
        self.learned_attn = False
        self.augment = False
        self.T = 0.07
        self.lambda_spatial = 10.0
        self.lambda_spatial_idt = 0.0
        self.lambda_perceptual = 0.0
        self.lambda_style = 0.0
        self.lambda_identity = 0.0
        self.lambda_gradient = 0.0

        self.gan_mode = "lsgan"
        self.pool_size = 50
        self.loss_names = ["style", "G_s", "per", "D_real", "D_fake", "G_GAN"]
        # specify the images you want to save/display
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk
        self.model_names = ["G", "D"] if self.isTrain else ["G"]

        self.norm_cfg = norm_cfg or {'type': 'in'}

        ###########################################################
        self.G = Generator(norm_cfg=self.norm_cfg).to(self.device)

        if self.isTrain:
            self.D = Discriminator().to(self.device)

            self.attn_layers = [int(i) for i in self.attn_layers.split(",")]

            if self.lambda_identity > 0.0 or self.lambda_spatial_idt > 0.0:
                # only works when input and output images have the
                # same number of channels
                self.visual_names.append("idt_B")
                if self.lambda_identity > 0.0:
                    self.loss_names.append("idt_B")
                if self.lambda_spatial_idt > 0.0:
                    self.loss_names.append("G_s_idt_B")

            if self.lambda_gradient > 0.0:
                self.loss_names.append("D_Gradient")
            self.fake_B_pool = ImagePool(
                self.pool_size
            )  # create image buffer to store previously generated images

            # define the loss function
            self.netPre = VGG16().to(self.device)
            self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
            self.criterionIdt = nn.L1Loss()
            self.criterionStyle = StyleLoss().to(self.device)
            self.criterionFeature = PerceptualLoss().to(self.device)
            self.criterionSpatial = SpatialCorrelativeLoss(
                self.loss_mode,
                self.patch_nums,
                self.patch_size,
                self.use_norm,
                self.learned_attn,
                T=self.T,
            ).to(self.device)
            self.normalization = Normalization(self.device)

            if self.learned_attn:
                self.F = self.criterionSpatial
                self.model_names.append("F")
                self.loss_names.append("spatial")
            else:
                self.set_requires_grad([self.netPre], False)
            # initialize optimizers
            self.optimizer_G = optim.Adam(
                itertools.chain(self.G.parameters()),
                lr=0.0001,
                betas=(0.5, 0.999),
            )
            self.optimizer_D = optim.Adam(
                itertools.chain(self.D.parameters()),
                lr=0.0001,
                betas=(0.5, 0.999),
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def Spatial_Loss(self, net, src, tgt, other=None):
        """
        given the source and target images to calculate
        the spatial similarity and dissimilarity loss
        """
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(
                torch.flip(other, [2, 3]), self.attn_layers, encode_only=True
            )
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(
            zip(feats_src, feats_tgt, feats_oth)
        ):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary
        pre-processing steps
        :param input:
        include the data itself and its metadata information
        :return:
        """
        self.real_A = input["X_img"].to(self.device)
        self.real_B = input["Y_img"].to(self.device)
        if self.isTrain and self.augment:
            self.aug_A = input["X_img_aug"].to(self.device)
            self.aug_B = input["Y_img_aug"].to(self.device)

    def forward(self):
        """Run forward pass"""
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if (
                self.lambda_identity + self.lambda_spatial_idt > 0
            ) and self.isTrain
            else self.real_A
        )
        self.fake = self.G(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if (
            self.lambda_identity + self.lambda_spatial_idt > 0
        ) and self.isTrain:
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_F(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        norm_real_A, norm_real_B, norm_fake_B = (
            self.normalization((self.real_A + 1) * 0.5),
            self.normalization((self.real_B + 1) * 0.5),
            self.normalization((self.fake_B.detach() + 1) * 0.5),
        )
        if self.augment:
            norm_aug_A, norm_aug_B = self.normalization(
                (self.aug_A + 1) * 0.5
            ), self.normalization((self.aug_B + 1) * 0.5)
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)
        self.loss_spatial = self.Spatial_Loss(
            self.netPre, norm_real_A, norm_fake_B, norm_real_B
        )

        self.loss_spatial.backward()

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: discriminator loss
        """
        # real
        real.requires_grad_()
        pred_real = netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_dis=True)
        # fake
        pred_fake = netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_dis=True)
        # combined loss
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        # gradient penalty
        if self.lambda_gradient > 0.0:
            self.loss_D_Gradient, _ = cal_gradient_penalty(
                netD, real, fake, real.device, lambda_gp=self.lambda_gradient
            )  #
            loss_D += self.loss_D_Gradient
        loss_D.backward()

        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(
            self.D, self.real_B, fake_B.detach()
        )

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        l_style = self.lambda_style
        l_per = self.lambda_perceptual
        l_sptial = self.lambda_spatial
        l_idt = self.lambda_identity
        l_spatial_idt = self.lambda_spatial_idt
        # GAN loss
        self.loss_G_GAN = self.criterionGAN(self.D(self.fake_B), True)
        # different structural loss
        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B + 1) * 0.5)
        self.loss_style = (
            self.criterionStyle(norm_real_B, norm_fake_B) * l_style
            if l_style > 0
            else 0
        )
        self.loss_per = (
            self.criterionFeature(
                norm_real_A, norm_fake_B
            ) * l_per if l_per > 0 else 0
        )
        self.loss_G_s = (
            self.Spatial_Loss(
                self.netPre, norm_real_A, norm_fake_B, None
            ) * l_sptial
            if l_sptial > 0
            else 0
        )
        # identity loss
        if l_spatial_idt > 0:
            norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            self.loss_G_s_idt_B = (
                self.Spatial_Loss(
                    self.netPre, norm_real_B, norm_fake_idt_B, None
                ) * l_spatial_idt
            )
        else:
            self.loss_G_s_idt_B = 0
        self.loss_idt_B = (
            self.criterionIdt(
                self.real_B, self.idt_B
            ) * l_idt if l_idt > 0 else 0
        )

        self.loss_G = (
            self.loss_G_GAN
            + self.loss_style
            + self.loss_per
            + self.loss_G_s
            + self.loss_G_s_idt_B
            + self.loss_idt_B
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        if self.learned_attn:
            self.set_requires_grad([self.F, self.netPre], True)
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()
        # D_A
        self.set_requires_grad([self.D], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # G_A
        self.set_requires_grad([self.D], False)
        self.optimizer_G.zero_grad()
        if self.learned_attn:
            self.set_requires_grad([self.F, self.netPre], False)
        self.backward_G()
        self.optimizer_G.step()

    def data_dependent_initialize(self, data):
        """
        The learnable spatially-correlative map is defined in terms
        of the shape of the intermediate, extracted features
        of a given network (encoder or pretrained VGG16).
        Because of this, the weights of spatial are initialized at the
        first feedforward pass with some input images
        :return:
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.isTrain:
            self.backward_G()
            self.optimizer_G.zero_grad()
            if self.learned_attn:
                self.optimizer_F = optim.Adam(
                    [
                        {
                            "params": list(
                                filter(
                                    lambda p: p.requires_grad,
                                    self.netPre.parameters()
                                )
                            ),
                            "lr": 0.0001 * 0.0,
                        },
                        {
                            "params": list(
                                filter(
                                    lambda p: p.requires_grad,
                                    self.F.parameters()
                                )
                            )
                        },
                    ],
                    lr=0.0001,
                    betas=(0.5, 0.999),
                )
                self.optimizers.append(self.optimizer_F)
                self.optimizer_F.zero_grad()

    def setup(self):
        """
        Create schedulers
        """
        if self.isTrain:
            lambda_lr = lambda epoch: 1.0 - max(
                0, epoch - self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2
            ) / (self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2)
            self.schedulers = [
                lr_scheduler.LambdaLR(optimizer, lambda_lr)
                for optimizer in self.optimizers
            ]

    def analyze_feature_map(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake, feature_map = self.G.analyze_feature_map(X)
        return Y_fake, feature_map

    def inference(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G(X)
        return Y_fake

    def inference_with_anchor(self, X, y_anchor, x_anchor):
        assert self.norm_cfg['type'] == "kin"
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G.forward_with_anchor(
                X, y_anchor=y_anchor, x_anchor=x_anchor,
            )
        return Y_fake

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def init_thumbnail_instance_norm_for_whole_model(self):
        init_thumbnail_instance_norm(self.G)

    def use_thumbnail_instance_norm_for_whole_model(self):
        use_thumbnail_instance_norm(self.G)

    def not_use_thumbnail_instance_norm_for_whole_model(self):
        not_use_thumbnail_instance_norm(self.G)

    def init_kernelized_instance_norm_for_whole_model(
        self, y_anchor_num, x_anchor_num
    ):
        init_kernelized_instance_norm(
            self.G,
            y_anchor_num=y_anchor_num,
            x_anchor_num=x_anchor_num,
        )

    def use_kernelized_instance_norm_for_whole_model(self):
        use_kernelized_instance_norm(self.G)

    def not_use_kernelized_instance_norm_for_whole_model(self):
        not_use_kernelized_instance_norm(self.G)
