import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.util import ReplayBuffer, weights_init

from models.base import BaseModel
from models.discriminator import Discriminator
from models.generator import Generator
from models.kin import (init_kernelized_instance_norm,
                        not_use_kernelized_instance_norm,
                        use_kernelized_instance_norm)
from models.projector import Head
from models.tin import (init_thumbnail_instance_norm,
                        not_use_thumbnail_instance_norm,
                        use_thumbnail_instance_norm)


class CycleGanModel(BaseModel):
    # instance normalization can be different from the one specified during training
    def __init__(self, config, normalization="in"):
        BaseModel.__init__(self, config)
        self.model_names = ['G_X2Y', 'G_Y2X', 'D_X', 'D_Y']
        self.loss_names = ['G_adv', 'D_Y', 'G', 'NCE']
        self.visual_names = ['X', 'Y', 'identity_X', 'identity_Y', 'fake_X', 'fake_Y', 'recovered_X', 'recovered_Y']
        
        self.normalization = normalization
        # Discrimnator would not be used during inference, 
        # so specification of instane normalization is not required
        self.G_X2Y = Generator().to(self.device)
        self.G_Y2X = Generator().to(self.device) 
        self.D_X = Discriminator(avg_pooling=True).to(self.device)
        self.D_Y = Discriminator(avg_pooling=True).to(self.device)

        self.G_X2Y.apply(weights_init)
        self.G_Y2X.apply(weights_init)
        self.D_X.apply(weights_init)
        self.D_Y.apply(weights_init)

        self.opt_G = optim.Adam(itertools.chain(self.G_X2Y.parameters(), self.G_Y2X.parameters()), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999))
        self.opt_D_X = torch.optim.Adam(self.D_X.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999))
        self.opt_D_Y = torch.optim.Adam(self.D_Y.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999))

        self.self.cycle_loss = nn.L1Loss().to(self.device)
        self.identity_loss = nn.L1Loss().to(self.device)
        self.self.adversarial_loss = nn.MSELoss().to(self.device)

        if self.config["TRAINING_SETTING"]["LOAD_MODEL"]:
            self.load_networks(self.config["TRAINING_SETTING"]["EPOCH"])

        lambda_lr = lambda epoch: 1.0 - max(0, epoch - self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2) / (self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2)
        self.scheduler_G = lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lambda_lr)
        self.scheduler_D_X = lr_scheduler.LambdaLR(self.opt_D_X, lr_lambda=lambda_lr)
        self.scheduler_D_Y = lr_scheduler.LambdaLR(self.opt_D_Y, lr_lambda=lambda_lr)

        self.fake_X_buffer = ReplayBuffer()
        self.fake_Y_buffer = ReplayBuffer()

    def set_input(self, input):
        self.X, self.Y = input

    def analyze_feature_map(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake, feature_map = self.G_X2Y.analyze_feature_map(X)
        return Y_fake, feature_map
    
    def inference(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G_X2Y(X)
        return Y_fake

    def inference_with_anchor(self, X, y_anchor, x_anchor, padding):
        assert self.normalization == "kin"
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G_X2Y.forward_with_anchor(X, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding)
        return Y_fake

    def optimize_parameters(self):
        X = self.X.to(self.device)
        Y = self.Y.to(self.device)
        batch_size = X.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=self.device, dtype=torch.float32)

        ##############################################
        # (1) Update G network: Generators X2Y and Y2X
        ##############################################
        # Set G_A and G_B's gradients to zero
        self.opt_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        self.identity_X = self.G_Y2X(X)
        loss_identity_X = self.identity_loss(self.identity_X, X) * 5.0
        # G_A2B(B) should equal B if real B is fed
        self.identity_Y = self.G_X2Y(Y)
        loss_identity_Y = self.identity_loss(self.identity_Y, Y) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        self.fake_X = self.G_Y2X(Y)
        fake_output_X = self.D_X(self.fake_X)
        loss_GAN_Y2X = self.adversarial_loss(fake_output_X, real_label)
        # GAN loss D_B(G_B(B))
        self.fake_Y = self.G_X2Y(X)
        fake_output_Y = self.D_Y(self.fake_Y)
        loss_GAN_X2Y = self.adversarial_loss(fake_output_Y, real_label)
        
        # Cycle loss
        self.recovered_X = self.G_Y2X(self.fake_Y)
        loss_cycle_XYX = self.cycle_loss(self.recovered_X, X) * 10.0

        self.recovered_Y = self.G_X2Y(self.fake_X)
        loss_cycle_YXY = self.cycle_loss(self.recovered_Y, Y) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_X + loss_identity_Y + loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_XYX + loss_cycle_YXY

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        self.opt_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        self.opt_D_X.zero_grad()

        # Real A image loss
        real_output_X = self.D_X(X)
        errD_real_X = self.adversarial_loss(real_output_X, real_label)

        # Fake A image loss
        self.fake_X = self.fake_X_buffer.push_and_pop(self.fake_X)
        fake_output_X = self.D_X(self.fake_X.detach())
        errD_fake_A = self.adversarial_loss(fake_output_X, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_X + errD_fake_A) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        self.opt_D_X.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        self.opt_D_Y.zero_grad()

        # Real B image loss
        real_output_B = self.D_Y(Y)
        errD_real_B = self.adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        self.fake_Y = self.fake_Y_buffer.push_and_pop(self.fake_Y)
        fake_output_Y = self.D_Y(self.fake_Y.detach())
        errD_fake_B = self.adversarial_loss(fake_output_Y, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        self.opt_D_Y.step()

    def scheduler_step(self):
        self.scheduler_G.step()
        self.scheduler_D_X.step()
        self.scheduler_D_Y.step()
    """
    def compute_D_loss(self):
        # Fake
        fake = self.Y_fake.detach()
        pred_fake = self.D_Y(fake)
        self.loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        self.pred_real = self.D_Y(self.Y)
        self.loss_D_real = self.mse(self.pred_real, torch.ones_like(self.pred_real))

        self.loss_D_Y = (self.loss_D_fake + self.loss_D_real) / 2
        return self.loss_D_Y

    def compute_G_loss(self):
        fake = self.Y_fake
        pred_fake = self.D_Y(fake)
        self.loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        self.loss_NCE = self.calculate_NCE_loss(self.X, self.Y_fake)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.Y, self.Y_idt)
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_Y) * 0.5

        self.loss_G = self.loss_G_adv + self.loss_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss
    """
    def init_thumbnail_instance_norm_for_whole_model(self):
        init_thumbnail_instance_norm(self.G)

    def use_thumbnail_instance_norm_for_whole_model(self):
        use_thumbnail_instance_norm(self.G)
    
    def not_use_thumbnail_instance_norm_for_whole_model(self):
        not_use_thumbnail_instance_norm(self.G)

    def init_kernelized_instance_norm_for_whole_model(self, y_anchor_num, x_anchor_num, kernel=(torch.ones(1,1,3,3)/9)):
        init_kernelized_instance_norm(
            self.G, 
            y_anchor_num=y_anchor_num, 
            x_anchor_num=x_anchor_num, 
            kernel=kernel
        )

    def use_kernelized_instance_norm_for_whole_model(self, padding=1):
        use_kernelized_instance_norm(self.G, padding=padding)
    
    def not_use_kernelized_instance_norm_for_whole_model(self):
        not_use_kernelized_instance_norm(self.G)
