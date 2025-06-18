#!/usr/bin/python3
import glob
import os
import argparse
import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt
from torch.nn import DataParallel
from models.counter_infer import build_model
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad
import torchvision.ops as ops
import colorsys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Count:
    def __init__(self, args):
        args.zero_shot = True
        self.device = torch.device("cuda")
        model = DataParallel(build_model(args).to(self.device))
        model.load_state_dict(torch.load('CNTQG_multitrain_ca44.pth', weights_only=True)['model'], strict=False)
        model.eval()
        self.model = model

    # **Post-process and Update Output**
    def post_process(self, image, outputs, img, scale, threshold):
        print('post process', threshold)
        idx = 0
        keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() * threshold],
                    outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() * threshold], 0.5)

        pred_boxes = outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() * threshold][keep]
        pred_boxes = torch.clamp(pred_boxes, 0, 1)

        pred_boxes = (pred_boxes / scale * img.shape[-1]).tolist()

        image = Image.fromarray((image).astype(np.uint8)).convert("RGBA")
        
        del outputs

        width, height = image.size
        draw = ImageDraw.Draw(image)
        for box in pred_boxes:
            draw.rectangle(box, outline="orange", width=width//250)

        square_size = int(0.05 * width)
        x1, y1 = 10, height - square_size - 10
        x2, y2 = x1 + square_size, y1 + square_size

        draw.rectangle([x1, y1, x2, y2], outline="black", fill="black", width=1)
        font = ImageFont.load_default(size=square_size)
        txt = str(len(pred_boxes))
        w = draw.textlength(txt, font=font)
        text_x = x1 + (square_size - w) / 2
        text_y = y1 - square_size/10
        draw.text((text_x, text_y), txt, fill="white", font=font, stroke_fill="white", stroke_width=1)

        return np.array(image.convert("RGB"), dtype=np.uint8)

    def predict(self, image, drawn_boxes, threshold):
        image_tensor = torch.tensor(image).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

        bboxes_tensor = torch.tensor(drawn_boxes, dtype=torch.float32).to(self.device)

        img, bboxes, scale = resize_and_pad(image_tensor, bboxes_tensor, size=1024.0)
        img = img.unsqueeze(0).to(self.device)
        bboxes = bboxes.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, _, _, _, _ = self.model(img, bboxes)
        
        outputs[0]['pred_boxes'] = outputs[0]['pred_boxes'].cpu()
        outputs[0]['box_v'] = outputs[0]['box_v'].cpu()

        torch.cuda.empty_cache()
        return self.post_process(image, outputs, img, scale, threshold)


class FolderProcessing:
    def __init__(self, method, args):
        folder = args.model_path
        self.img_list = glob.iglob(os.path.join(folder, '*.png'))
        self.img_list = sorted(self.img_list)
        self.folder = folder
        self.method = method(args)

    def run(self):
        print(self.img_list)
        import cv2
        for img_filename in self.img_list:

            frame = cv2.imread(img_filename)
            frame = self.method.predict(frame)

            if self.folder is not None:
                cv2.imwrite(os.path.basename(img_filename), frame)
            else:
                import pylab as plt
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.show(block=True)


def main(args):
    if args.image_folder is None:
        from echolib_wrapper import EcholibWrapper
        p = EcholibWrapper(Count, args)
    else:
        p = FolderProcessing(Count, args)

    try:
        p.run()
    except Exception as ex:
        print(ex)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('GeCo2', parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
