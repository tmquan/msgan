"""
To run this template just do:
python generative_adversarial_net.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os
import copy
from argparse import ArgumentParser
from collections import OrderedDict

import random
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn.functional import avg_pool2d
        
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer


#
import tensorpack.dataflow as df
from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData
from tensorpack.utils import get_rng
from tensorpack.utils.argtools import shape2d

import albumentations as AB
import sklearn.metrics
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
class MultiLabelDataset(df.RNGDataFlow):
    def __init__(self, folder, types=14, is_train='train', channel=1,
                 resize=None, debug=False, shuffle=False, pathology=None, 
                 fname='train.csv', balancing=None):

        self.version = "1.0.0"
        self.description = "Vinmec is a large dataset of chest X-rays\n",
        self.citation = "\n"
        self.folder = folder
        self.types = types
        self.is_train = is_train
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if self.channel == 1:
            self.imread_mode = cv2.IMREAD_GRAYSCALE
        else:
            self.imread_mode = cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.debug = debug
        self.shuffle = shuffle
        self.csvfile = os.path.join(self.folder, fname)
        print(self.folder)
        # Read the csv
        self.df = pd.read_csv(self.csvfile)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df = self.df.infer_objects()
        
        self.pathology = pathology
        self.balancing = balancing
        if self.balancing == 'up':
            self.df_majority = self.df[self.df[self.pathology]==0]
            self.df_minority = self.df[self.df[self.pathology]==1]
            print(self.df_majority[self.pathology].value_counts())
            self.df_minority_upsampled = resample(self.df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=self.df_majority[self.pathology].value_counts()[0],    # to match majority class
                                     random_state=123) # reproducible results

            self.df_upsampled = pd.concat([self.df_majority, self.df_minority_upsampled])
            self.df = self.df_upsampled
    def reset_state(self):
        self.rng = get_rng(self)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.is_train == 'train':
            self.rng.shuffle(indices)

        for idx in indices:
            fpath = os.path.join(self.folder, 'data')
            fname = os.path.join(fpath, self.df.iloc[idx]['Images'])
            image = cv2.imread(fname, self.imread_mode)
            assert image is not None, fname
            # print('File {}, shape {}'.format(fname, image.shape))
            if self.channel == 3:
                image = image[:, :, ::-1]
            if self.resize is not None:
                image = cv2.resize(image, tuple(self.resize[::-1]))
            if self.channel == 1:
                image = image[:, :, np.newaxis]

            # Process the label
            if self.is_train == 'train' or self.is_train == 'valid':
                label = []
                if self.types == 6:
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Fracture'])
                    label.append(self.df.iloc[idx]['Lung_Lesion'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    label.append(self.df.iloc[idx]['Pneumothorax'])
                if self.types == 4:
                    label.append(self.df.iloc[idx]['Covid'])
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Consolidation'])
                    label.append(self.df.iloc[idx]['Pneumonia'])
                elif self.types == 2:
                    assert self.pathology is not None
                    label.append(self.df.iloc[idx]['No_Finding'])
                    label.append(self.df.iloc[idx][self.pathology])
                else:
                    pass
                # Try catch exception
                label = np.nan_to_num(label, copy=True, nan=0)
                label = np.array(label>0.16, dtype=np.float32)
                types = label.copy()
                yield [image, types]
            elif self.is_train == 'test':
                yield [image] 
            else:
                pass


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def DiceScore(output, target, smooth=1.0, epsilon=1e-7, axis=(2, 3)):
    """
    Compute mean dice coefficient over all abnormality classes.
    Args:
        output (Numpy tensor): tensor of ground truth values for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        target (Numpy tensor): tensor of predictions for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    y_true = target
    y_pred = output
    dice_numerator = 2*np.sum(y_true*y_pred, axis=axis) + epsilon
    dice_denominator = (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + epsilon)
    dice_coefficient = np.mean(dice_numerator / dice_denominator)

    return dice_coefficient

class SoftDiceLoss(nn.Module):
    def init(self):
        super(SoftDiceLoss, self).init()

    def forward(self, output, target, smooth=1.0, epsilon=1e-7, axis=(1)):
        """
        Compute mean soft dice loss over all abnormality classes.
        Args:
            y_true (Torch tensor): tensor of ground truth values for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            y_pred (Torch tensor): tensor of soft predictions for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                          denominator in formula for dice loss.
                          Hint: pass this as the 'axis' argument to the K.sum
                                and K.mean functions.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_loss (float): computed value of dice loss.  
        """
        y_true = target
        y_pred = output
        dice_numerator = 2*torch.sum(y_true*y_pred, dim=axis) + epsilon
        dice_denominator = (torch.sum(y_true*y_true, dim=axis) + torch.sum(y_pred*y_pred, dim=axis) + epsilon)
        dice_coefficient = torch.mean(dice_numerator / dice_denominator)
        
        dice_loss = 1 - dice_coefficient
        return dice_loss

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_size = self.hparams.shape // 16  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.hparams.latent_dim+self.hparams.types, 1024 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            # nn.Conv2d(1024, 4*512, 3, stride=1, padding=1),
            # nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            # nn.Conv2d(512, 4*256, 3, stride=1, padding=1),
            # nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            # nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            # nn.Conv2d(256, 4*128, 3, stride=1, padding=1),
            # nn.PixelShuffle(2),
            # nn.Dropout(0.25),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            # nn.Conv2d(128, 4*64, 3, stride=1, padding=1),
            # nn.PixelShuffle(2),
            # nn.Dropout(0.25),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # if self.hparams.arch.lower() == 'densenet121':
        self.discrim = getattr(torchvision.models, 'densenet121')(
            pretrained=True)
        self.discrim.features.conv0 = nn.Conv2d(1, 64, 
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.discrim.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Identity(),
            # nn.Linear(1024, self.hparams.types),  # 5 diseases
            # nn.Sigmoid(),
        )
        print(self.discrim)

        self.adv_layer = nn.Sequential(nn.Linear(1024, self.hparams.types), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(1024, self.hparams.types), nn.Sigmoid())

    def forward(self, img):
        # out = self.conv_blocks(img)
        # out = out.view(out.shape[0], -1)
        out = self.discrim(img)
        pred = self.adv_layer(out)
        prob = self.aux_layer(out)

        return pred, prob

class SGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device("cuda")
    
        self.gen = Generator(hparams).to(self.device)
        self.dis = Discriminator(hparams).to(self.device)
        self.gen.apply(weights_init_normal)
        self.dis.apply(weights_init_normal)

        # self.loss_fn = LSGAN(self.dis)
        # self.loss_fn = RelativisticAverageHingeGAN(self.dis)
        # self.loss_fn = StandardGAN(self.dis)
        # self.loss_fn = LSGAN_SIGMOID(self.dis)

        self.adversarial_loss = SoftDiceLoss() #torch.nn.BCELoss()
        self.probability_loss = SoftDiceLoss() #torch.nn.BCELoss()

        # cache for generated images
        self.fake_imgs = None
        self.real_imgs = None

    def forward(self, z):
        # return self.gen(z)
        true_or_fake, prob = self.dis(x)
        return prob


    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, lbls = batch
        imgs = imgs.to(self.device) / 128.0 - 1.0
        lbls = lbls.to(self.device)
        self.real_imgs = imgs

        batchs = imgs.shape[0]
        # sample some random latent points
        n = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        p = torch.empty(batchs, self.hparams.types).random_(2).to(self.device)

        z = torch.cat([n, p*2-1], dim=1)

        fake_imgs = self.gen(z)

        # train generator
        if optimizer_idx == 0:
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            fake = torch.ones_like(p)

            # adversarial loss is binary cross-entropy
            pred, prob = self.dis(fake_imgs)
            g_loss = (self.adversarial_loss(pred, fake) +  self.probability_loss(prob, p)) 
            # g_loss = (-torch.mean(pred) +  self.probability_loss(prob, p)) 

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train dis
        if optimizer_idx == 1:
            # how well can it label as real?
            true = torch.ones_like(p)
            real_pred, real_prob = self.dis(imgs)
            real_loss = (self.adversarial_loss(real_pred, true) +  self.probability_loss(real_prob, lbls)) 
            # real_loss = (-torch.mean(real_pred) +  self.probability_loss(real_prob, lbls)) 

            # how well can it label as fake?
            fake = torch.zeros_like(p)
            fake_pred, fake_prob = self.dis(fake_imgs) #.detach()
            fake_loss = (self.adversarial_loss(fake_pred, fake) +  self.probability_loss(fake_prob, p)) 
            # fake_loss = (torch.mean(fake_pred) +  self.probability_loss(fake_prob, p)) 

            # dis loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        ds_train = MultiLabelDataset(folder=self.hparams.data,
                                     is_train='train',
                                     fname='covid_train_v5.csv',
                                     types=self.hparams.types,
                                     pathology=self.hparams.pathology,
                                     resize=int(self.hparams.shape),
                                     balancing=None)

        ds_train.reset_state()
        ag_train = [
            # imgaug.Albumentations(
            #     AB.SmallestMaxSize(self.hparams.shape, p=1.0)),
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            # imgaug.Affine(shear=10),
            imgaug.RandomChooseAug([
                imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),
            ]),
            imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), p=0.5)),
            imgaug.RandomOrderAug([
                imgaug.Affine(shear=10, border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(translate_frac=(0.01, 0.02), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(scale=(0.5, 1.0), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
            ]),
            imgaug.RotationAndCropValid(max_deg=10, interp=cv2.INTER_AREA),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_AREA, target_shape=self.hparams.shape),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        # Label smoothing
        ag_label = [
            imgaug.BrightnessScale((0.8, 1.2), clip=False),
        ]
        # ds_train = AugmentImageComponent(ds_train, ag_label, 1)
        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                       torch.tensor(dp[1]).float()])
        return ds_train

    def on_epoch_end(self):
        batchs = 16
        n = torch.randn(batchs, self.hparams.latent_dim).to(self.device)
        p = torch.empty(batchs, self.hparams.types).random_(2).to(self.device)

        z = torch.cat([n, p*2-1], dim=1)
        # log sampled images
        self.fake_imgs = self.gen(z)

        grid = torchvision.utils.make_grid(self.fake_imgs[:batchs] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'fake_imgs', grid, self.current_epoch)

        grid = torchvision.utils.make_grid(self.real_imgs[:batchs] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'real_imgs', grid, self.current_epoch)

        self.viz = nn.Sequential(
            self.gen,
            self.dis
            )
        self.logger.experiment.add_graph(self.viz, z)



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hparams.load:
        model = SGAN(hparams).load_from_checkpoint(hparams.load)
    else:
        model = SGAN(hparams)

    custom_log_dir = os.path.join(str(hparams.save),
                                  str(hparams.pathology),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  # str(hparams.folds),
                                  # str(hparams.valid_fold_index)
                                  ),

    # checkpoint_callback = ModelCheckpoint(
    #     filepath=custom_log_dir,
    #     verbose=True,
    #     # filepath=os.path.join(custom_log_dir, 'ckpt'),
    #     # save_top_k=10,
    #     # monitor='val_f1_score_0',  # TODO
    #     # mode='max'
    # )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = Trainer()
   
    trainer = Trainer(
        num_sanity_val_steps=0,
        default_root_dir=os.path.join(str(hparams.save),
                                  str(hparams.pathology),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  # str(hparams.folds),
                                  # str(hparams.valid_fold_index)
                                  ),
        default_save_path=os.path.join(str(hparams.save),
                                  str(hparams.pathology),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  # str(hparams.folds),
                                  # str(hparams.valid_fold_index)
                                  ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        # checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        # train_percent_check=hparams.percent_check,
        # val_percent_check=hparams.percent_check,
        # test_percent_check=hparams.percent_check,
        # distributed_backend=hparams.distributed_backend,
        # use_amp=hparams.use_16bit,
        # val_check_interval=hparams.val_check_interval,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # trainer.fit(model)
    if hparams.eval:
        assert hparams.loadD
        model.eval()
        # trainer.test(model)
        pass
    elif hparams.pred:
        assert hparams.load
        model.eval()
        pass
    else:
        trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Training params
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument('--percent_check', type=float, default=0.1)

    parser.add_argument('--data', metavar='DIR', default=".", type=str, help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str, help='path to save output')
    parser.add_argument('--info', metavar='DIR', default="train_log", help='path to logging output')
    parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    parser.add_argument('--seed', type=int, default=1, help='reproducibility')

    # Inference params
    parser.add_argument('--load', action='store_true', help='path to logging output')
    parser.add_argument('--pred', action='store_true', help='run predict')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')

    # Dataset params
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pathology', default='Covid')
    parser.add_argument('--types', type=int, default=1)
    parser.add_argument('--shape', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--debug', action='store_true', help='use fast mode')
    
    hparams = parser.parse_args()

    main(hparams)