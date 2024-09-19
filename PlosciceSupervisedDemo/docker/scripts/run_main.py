#!/usr/bin/python3

import numpy as np

import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F

import cv2

cv2.ocl.setUseOpenCL(False)

import torch
from torchvision import transforms as T
from PIL import Image

from utils import crop_part_multi

print("Ploscice startup")

DEVICE = "cuda:0"
RESIZE = (256, 256)

CROP_PIXEL = 5              # ignore 5px border when object is cropped
MIN_BOARD_AREA = 200*200    # object should be at least 200x200 px in area

THRESHOLD = 0.50
TEXT_THICKNES = 7

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0, 1)
        nn.init.zeros_(self.bias)

class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())


class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


class SegDecNet(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNet, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.volume = nn.Sequential(_conv_block(self.input_channels, 32, 5, 2),
                                    # _conv_block(32, 32, 5, 2), # Has been accidentally left out and remained the same since then
                                    nn.MaxPool2d(2),
                                    _conv_block(32, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2),
                                    _conv_block(64, 1024, 15, 7))

        self.seg_mask = nn.Sequential(
            Conv2d_init(in_channels=1024, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))

        self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1025, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))

        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        self.fc = nn.Linear(in_features=66, out_features=1)

        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        self.device = device


    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        volume = self.volume(input)
        seg_mask = self.seg_mask(volume)

        cat = torch.cat([volume, seg_mask], dim=1)

        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)

        features = self.extractor(cat)
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        return prediction, seg_mask


class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None
    

class SegDecNetModel:

    def __init__(self, modelFile):
        self.model = SegDecNet(DEVICE, RESIZE[0], RESIZE[1], input_channels=3)
        self.model.set_gradient_multipliers(1)
        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(modelFile, map_location=DEVICE))

        self.model.eval()

    def predict(self, image):
        try:
        
            drawn_true = False
            drawn_contours = image.copy()
            for final_cropped, contours, tx, ty in crop_part_multi(image, CROP_PIXEL, MIN_AREA_THR=MIN_BOARD_AREA):

                if final_cropped is None:
                    continue
                
                print(final_cropped.shape, final_cropped.shape[0]*final_cropped.shape[1], MIN_BOARD_AREA*MIN_BOARD_AREA)

                final_cropped = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)

                # image_t = T.Compose([T.Resize(RESIZE, Image.ANTIALIAS), T.ToTensor(),                             T.Normalize(mean=imagenet_mean, std=imagenet_std)])(Image.fromarray(image_cv2)).unsqueeze(0).to(DEVICE)
                image_t = T.Compose([T.Resize(RESIZE, Image.ANTIALIAS), T.ToTensor()])(Image.fromarray(final_cropped)).unsqueeze(0).to(DEVICE)

                pred,pred_mask = self.model(image_t)
                pred = torch.sigmoid(pred)
                pred_mask = torch.sigmoid(pred_mask)

                image_score = pred.item()

                if image_score < THRESHOLD:
                    c = [0, 255, 0]
                    drawn_contours = cv2.drawContours(drawn_contours, contours, 0, c, 20)
                    drawn_contours = cv2.putText(drawn_contours, f"OK ({image_score * 100:.1f})", (tx-220, ty+500), cv2.FONT_HERSHEY_TRIPLEX, 4, c, TEXT_THICKNES)
                else:
                    c = [0, 0, 255]
                    drawn_contours = cv2.drawContours(drawn_contours, contours, 0, c, 20)
                    drawn_contours = cv2.putText(drawn_contours, f"X ({image_score * 100:.1f})", (tx-220, ty+500), cv2.FONT_HERSHEY_TRIPLEX, 4, c, TEXT_THICKNES)
                
                drawn_true = True
            
            if not drawn_true:
                raise Exception("No objects found")

        except Exception as e:
            import traceback 
            traceback.print_exc() 
            print(e)

            cv2.putText(drawn_contours, f"X", (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, [0, 0, 255], 5)

        return cv2.cvtColor(drawn_contours, cv2.COLOR_BGR2RGB)


import glob, os

class FolderProcessing:
    def __init__(self, detection_method, folder, img_ext, out_folder=None):
        self.detection_method = detection_method
        self.img_list = glob.iglob(os.path.join(folder, '*.' + img_ext))
        self.img_list = sorted(self.img_list)

        self.out_folder = out_folder

    def run(self):
        for img_filename in self.img_list:

            frame = cv2.imread(img_filename)
            frame = self.detection_method.predict(frame)

            if self.out_folder != None:
                cv2.imwrite(os.path.join(self.out_folder, os.path.basename(img_filename)), frame)
            else:
                import pylab as plt
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.show(block=True)


def main(args):
    if args.image_folder is None:
        from echolib_wrapper import EcholibWrapper
        processer = lambda d: EcholibWrapper(d)
    else:
        processer = lambda d: FolderProcessing(d, args.image_folder, args.image_ext, args.out_folder)

    p = processer(SegDecNetModel(modelFile=args.model))

    try:
        p.run()
    except Exception as ex:
        print(ex)
        pass


def parseArgs():
    parser = argparse.ArgumentParser(description='Tile defect segmentation')
    parser.add_argument(
        '--model',
        dest='model',
        help='model model file',
        default="model.pth",
        type=str
    )
    parser.add_argument(
        '--model_seg',
        dest='model_seg',
        help='model segmentation file',
        default="model_seg.pth",
        type=str
    )

    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--image-folder',
        dest='image_folder',
        help='folder to images for processing (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        help='folder to store output (default: None)',
        default=None,
        type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    main(args)
