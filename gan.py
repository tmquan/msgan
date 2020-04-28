"""
To run this template just do:
python generative_adversarial_net.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

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

from pytorch_lightning.core import LightningModule
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

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # networks
        shape = (1, hparams.shape, hparams.shape)
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=shape)
        self.discriminator = Discriminator(img_shape=shape)

        # cache for generated images
        self.fake_imgs = None
        self.real_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        imgs = imgs / 128.0 - 1.0
        self.real_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            z = z.type_as(imgs)

            # generate images
            self.fake_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.fake_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(fake)

            fake_loss = self.adversarial_loss(self.discriminator(self.fake_imgs.detach()), fake)

            # discriminator loss is the average of these
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

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize([0.5], [0.5])])
        # dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        # return DataLoader(dataset, batch_size=self.hparams.batch_size)
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
        z = torch.randn(16, self.hparams.latent_dim)
        z = z.type_as(self.real_imgs)

        # log sampled images
        self.fake_imgs = self(z)

        grid = torchvision.utils.make_grid(self.fake_imgs[:16] / 2.0 + 0.5)
        self.logger.experiment.add_image(f'fake_imgs', grid, self.current_epoch)

        grid = torchvision.utils.make_grid(self.real_imgs[:16] / 2.0 + 0.5)
        self.logger.experiment.add_image(f'real_imgs', grid, self.current_epoch)

        self.viz = nn.Sequential(
            self.generator,
            self.discriminator
            )
        self.logger.experiment.add_graph(self.viz, z)



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer()

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    parser.add_argument('--data', metavar='DIR', default=".", type=str, help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str, help='path to save output')
    parser.add_argument('--info', metavar='DIR', default="train_log", help='path to logging output')
    parser.add_argument('--gpus', type=int, default=1, help='how many gpus')

    parser.add_argument('--types', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pathology', default='All')
    parser.add_argument('--shape', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--debug', action='store_true', help='use fast mode')
    
    # Inference purpose
    parser.add_argument('--load', action='store_true', help='path to logging output')
    parser.add_argument('--pred', action='store_true', help='run predict')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    hparams = parser.parse_args()

    main(hparams)