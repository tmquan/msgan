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

# from CustomLayers import *
from Losses import *

class Generator(torch.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True, gpu_parallelize=False):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from CustomLayers import GenGeneralConvBlock, GenInitialBlock, _equalized_conv2d

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                return _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return Conv2d(in_channels, 1, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))

        return outputs

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return torch.clamp(data, min=0, max=1)


class Discriminator(torch.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from CustomLayers import DisGeneralConvBlock, DisFinalBlock, _equalized_conv2d
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return _equalized_conv2d(1, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return Conv2d(1, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = torch.nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = torch.nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = torch.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y


class MSGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # networks
        # shape = (1, hparams.shape, hparams.shape)
        # self.gen = Generator(latent_dim=hparams.latent_dim, img_shape=shape)
        # self.discriminator = Discriminator(img_shape=shape)
        depth = 7
        latent_size = hparams.latent_dim
        use_eql = True
        use_ema = True
        ema_decay = 0.999
        normalize_latents = False
        

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.normalize_latents = normalize_latents
        self.device = torch.device("cuda")

        self.gen = Generator(depth, latent_size, use_eql=use_eql, gpu_parallelize=True).to(self.device)
        self.dis = Discriminator(depth, latent_size, use_eql=use_eql, gpu_parallelize=True).to(self.device)

        # self.loss_fn = LSGAN(self.dis)
        self.loss_fn = RelativisticAverageHingeGAN(self.dis)

        if self.use_ema:
            from CustomLayers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)


        # cache for generated images
        self.fake_imgs = None
        self.real_imgs = None

    def forward(self, z):
        return self.gen(z)

    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device) / 128.0 - 1.0
        self.real_imgs = imgs

        batchs = imgs.shape[0]
        # create a list of downsampled images from the real images:
        images = [imgs] + [avg_pool2d(imgs, int(np.power(2, i))) for i in range(1, self.depth)]
        images = list(reversed(images))

        # sample some random latent points
        noises = torch.randn(batchs, self.latent_size).to(self.device)

        # normalize them if asked
        if self.normalize_latents:
            noises = (noises / noises.norm(dim=-1, keepdim=True) * (self.latent_size ** 0.5))

        fake_images = self.gen(noises)
        # train generator
        if optimizer_idx == 0:
            g_loss = self.loss_fn.gen_loss(images, fake_images)
            # g_loss =  0.5 * ((torch.mean(self.dis(fake_images)) - 1) ** 2)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            # if self.use_ema is true, apply the moving average here:
            if self.use_ema:
                self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)
            return output

        # train discriminator
        if optimizer_idx == 1:
            fake_images = list(map(lambda x: x.detach(), fake_images))
            d_loss = self.loss_fn.dis_loss(images, fake_images)
            # d_loss = 0.5 * (((torch.mean(self.dis(images)) - 1) ** 2)
            #                + (torch.mean(self.dis(fake_images))) ** 2)
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
        z = torch.randn(16, self.hparams.latent_dim).to(self.device)
        # z = z.type_as(self.real_imgs)

        # log sampled images
        self.fake_imgs = self(z)[-1]

        grid = torchvision.utils.make_grid(self.fake_imgs[:16] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'fake_imgs', grid, self.current_epoch)

        grid = torchvision.utils.make_grid(self.real_imgs[:16] / 2.0 + 0.5, normalize=True)
        self.logger.experiment.add_image(f'real_imgs', grid, self.current_epoch)

        # self.viz = nn.Sequential(
        #     self.gen,
        #     self.dis
        #     )
        # self.logger.experiment.add_graph(self.viz, z)



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = MSGAN(hparams)

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
    # Training params
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")

    parser.add_argument('--data', metavar='DIR', default=".", type=str, help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str, help='path to save output')
    parser.add_argument('--info', metavar='DIR', default="train_log", help='path to logging output')
    parser.add_argument('--gpus', type=int, default=1, help='how many gpus')

    # Inference params
    parser.add_argument('--load', action='store_true', help='path to logging output')
    parser.add_argument('--pred', action='store_true', help='run predict')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')

    # Dataset params
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pathology', default='All')
    parser.add_argument('--types', type=int, default=1)
    parser.add_argument('--shape', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--debug', action='store_true', help='use fast mode')
    
    hparams = parser.parse_args()

    main(hparams)