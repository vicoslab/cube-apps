#!/usr/bin/python3
import glob
import os
import argparse
import cv2
import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt

from dave.dave import build_model

cv2.ocl.setUseOpenCL(False)
def resize_and_pad(img, bboxes=None, size=512.0):
    bs, channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension
    resized_img = torch.nn.functional.interpolate(img, scale_factor=scaling_factor, mode='bilinear',
                                                     align_corners=False)
    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])
    padded_img = torch.nn.functional.pad(resized_img, (0, pad_width, 0, pad_height), mode='constant', value=0)[0]

    _, W, H = img.shape
    if bboxes is not None:
        bboxes = bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor])
        return padded_img, bboxes, scaling_factor
    else:
        return padded_img, scaling_factor

class Count:
    def __init__(self, args):
        gpu = 0
        torch.cuda.set_device(gpu)
        self.device = torch.device(gpu)
        # self.device = 'cpu'
        model = build_model(args).to(self.device)

        new_state = {k.split("module.")[1]: v for k, v in
                     torch.load(os.path.join(args.model_path, args.model_name + '.pth'))[
                         'model'].items()}

        model.load_state_dict(new_state, strict=False)
        pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                                torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                    'model'].items() if 'feat_comp' in k}
        model.feat_comp.load_state_dict(pretrained_dict_feat)

        model.eval()
        self.model = model

    def predict(self, image):
        with torch.no_grad():
            # transform image
            image_t = T.ToTensor()(image).unsqueeze(0)
            image_t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_t)
            image, scaling_factor = resize_and_pad(image)
            image_t = image_t.to(self.device)
            bboxes = torch.zeros((1, 3, 4), dtype=torch.float32, device=self.device)

            # predict bboxes and density maps
            denisty_map, _, _, predicted_bboxes = self.model(image_t, bboxes)

            # add bboxes to image
            for i in range(len(predicted_bboxes.box)):
                box = predicted_bboxes.box.cpu()[i].numpy()/scaling_factor
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)

            image = cv2.putText(image, "Dmap count:" + str(round(denisty_map.sum().item(), 1)) + " Box count:" + str(
                len(predicted_bboxes.box)), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1, cv2.LINE_AA)

            print("Dmap count:" + str(round(denisty_map.sum().item(), 1)) + " Box count:" + str(
                len(predicted_bboxes.box)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FolderProcessing:
    def __init__(self, method, folder):
        self.img_list = glob.iglob(os.path.join(folder, '*.png'))
        self.img_list = sorted(self.img_list)
        self.folder = folder
        self.method = method

    def run(self):
        print(self.img_list)
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
        processer = lambda d: EcholibWrapper(d)
    else:
        processer = lambda d: FolderProcessing(d, args.model_path)

    p = processer(Count(args))

    try:
        p.run()
    except Exception as ex:
        print(ex)
        pass


def get_argparser():
    parser = argparse.ArgumentParser("DAVE parser", add_help=False)
    parser.add_argument('--model_name', default='DAVE_0_shot', type=str)
    parser.add_argument('--data_path', default=r'C:\projects\DAVE\FSC147_384_V2', type=str)
    parser.add_argument('--model_path', default='material/', type=str)
    parser.add_argument('--det_model_name', default='DAVE', type=str)
    parser.add_argument('--dataset', default='fsc147', type=str)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--swav_backbone', action='store_true')
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--fcos_pred_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_dec_layers', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--detection_loss_weight', default=0.01, type=float)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--task', default='fscd147', type=str)
    parser.add_argument('--d_s', default=1.0, type=float)
    parser.add_argument('--m_s', default=0.0, type=float)
    parser.add_argument('--i_thr', default=0.55, type=float)
    parser.add_argument('--d_t', default=3, type=float)
    parser.add_argument('--s_t', default=0.008, type=float)
    parser.add_argument('--norm_s', action='store_true')
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--egv', default=0.132, type=float)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--det_train', action='store_true')
    parser.add_argument('--eval_multicat', action='store_true')
    parser.add_argument('--prompt_shot', action='store_true')
    parser.add_argument('--normalized_l2', action='store_true')
    parser.add_argument('--count_loss_weight', default=0, type=float)
    parser.add_argument('--min_count_loss_weight', default=0, type=float)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--use_query_pos_emb', action='store_true')
    parser.add_argument('--use_objectness', action='store_true')
    parser.add_argument('--use_appearance', action='store_true')
    parser.add_argument('--orig_dmaps', action='store_true')
    parser.add_argument('--skip_cars', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--image_folder', type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
