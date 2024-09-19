#!/usr/bin/python3

import numpy as np
import cv2

import sys
import argparse

cv2.ocl.setUseOpenCL(False)

import torch
from torchvision import transforms as T
from PIL import Image
from efficientnet_pytorch import EfficientNet

DEVICE = "cuda:0"
RESIZE = (1512, 536)

CROP_PIXEL = 5           # ignore 5px border when object is cropped
MIN_BOARD_AREA = 200*200 # object should be at least 200x200 px in area

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_part_single(image):
    final_crop = 20

    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 50
    tried = []
    while True:
        tried.append(threshold)
        r, t = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 25

        eroded = cv2.erode(t, np.ones(kernel_size))
        dilated = cv2.dilate(eroded, np.ones((kernel_size, kernel_size)))
        ccs = cv2.connectedComponentsWithStats(dilated)
        if ccs[0] > 2:
            threshold += 5
        elif ccs[0] < 2:
            threshold -= 1
        else:
            break
        if threshold in tried:
            print(f"Cant find suitable threshold!!!")
            return None, None, 0, 0

    ccs = cv2.connectedComponentsWithStats(dilated)

    i = 1
    x, y, w, h, area = ccs[2][i]
    cc_mask = ccs[1] == i
    contours, _ = cv2.findContours(image=cc_mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    _, (w_min, h_min), angle = cv2.minAreaRect(contours[0])

    points_bb_rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    print(points_bb_rect)
    bb_rect = cv2.fillPoly(np.zeros_like(bw), [points_bb_rect], (255))
    cutout = image[y:y + h, x:x + w]
    rotated = rotate_image(cutout, angle)

    h_r, w_r, _ = rotated.shape
    y_rc, x_rc = h_r // 2, w_r // 2
    final = rotated[int(y_rc - h_min // 2):int(y_rc + h_min // 2), int(x_rc - w_min // 2):int(x_rc + w_min // 2)]
    final_cropped = final[final_crop:-final_crop, final_crop:-final_crop]

    if final_cropped.shape[0] < final.shape[1]:
        final_cropped = cv2.rotate(final, cv2.ROTATE_90_CLOCKWISE)

    ty = int(y + h // 2) + 45
    tx = int(x + w // 2) - 45

    return final_cropped, contours, tx, ty


def crop_part_multi(image, final_crop, MIN_AREA_THR=100):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #################################################################
    # apply thresholding

    r, t = cv2.threshold(bw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(t,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    ccs = cv2.connectedComponentsWithStats(sure_bg)
    for i in range(len(ccs[2])):

        x, y, w, h, area = ccs[2][i]
        cc_mask = ccs[1] == i
        contours_all, _ = cv2.findContours(image=cc_mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
        _, (w_min, h_min), angle = cv2.minAreaRect(contours_all[0])


        if final_crop > 0:
            cutout = image[y:y + h, x:x + w]
        else:
            cutout = image[y + int(1.5 * final_crop):y + h - int(1.5 * final_crop), x + int(1.5 * final_crop):x + w - int(1.5 * final_crop)]
        rotated = rotate_image(cutout, angle)

        h_r, w_r, _ = rotated.shape
        y_rc, x_rc = h_r // 2, w_r // 2
        if final_crop > 0:
            final_cropped = rotated[int(y_rc - h_min // 2) + final_crop:int(y_rc + h_min // 2) - final_crop, int(x_rc - w_min // 2) + final_crop:int(x_rc + w_min // 2) - final_crop]
            final_cropped = final_cropped[final_crop:-final_crop, final_crop:-final_crop]
        else:
            crop = int((w_r - w_min + 2 * final_crop) / 2)
            final_cropped = rotated[crop:-crop, crop:-crop]

        # skip if contour is actually the whole image or small image (< MIN_AREA_THR px)
        if final_cropped.shape[0] * final_cropped.shape[1] < max(MIN_AREA_THR,1) or \
            final_cropped.shape[0] * final_cropped.shape[1] > image.shape[0]*image.shape[1]*0.95:
            continue

        if final_cropped.shape[0] < final_cropped.shape[1]:
            final_cropped = cv2.rotate(final_cropped, cv2.ROTATE_90_CLOCKWISE)

        ty = int(y + h // 2) + 45
        tx = int(x + w // 2) - 45

        yield final_cropped, contours_all, tx, ty


class PModel:

    def __init__(self, modelFile, blockNumber=4, sizeRange=None):
        self.model = EfficientNet.from_name('efficientnet-b4', num_classes=10).to(DEVICE)
        self.model.load_state_dict(torch.load(modelFile, map_location=DEVICE))
        self.model.eval()
        self.model.to(DEVICE)

    def predict(self, image):

        try:
            drawn_true = False
            drawn_contours = image.copy()            
            for cropped_part, contours, tx, ty in crop_part_multi(image, CROP_PIXEL, MIN_AREA_THR=MIN_BOARD_AREA):

                if cropped_part is None:
                    print("cropped_part is None")
                    continue
                
                print("Board shape: ",cropped_part.shape, MIN_BOARD_AREA, "tx,ty:", tx, ty)
                
                #if cropped_part.shape[0]*cropped_part.shape[1] < FILTER_SIZE*FILTER_SIZE:
                #    continue

                image_t = T.Compose([T.Resize(RESIZE, Image.ANTIALIAS), T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(Image.fromarray(cropped_part)).unsqueeze(0).to(DEVICE)
                predictions = self.model(image_t)
                predicted = predictions.argmax().item()

                print(f"Predicted: {predicted}")
                
                drawn_contours = cv2.drawContours(drawn_contours, contours, 0, [0, 255, 0], 20)
                drawn_contours = cv2.putText(drawn_contours, str(predicted), (tx, ty), cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 255, 0), 20)
                
                drawn_true = True

            if not drawn_true:
                raise Exception("No contours drawn")
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

    p = processer(PModel(modelFile=args.model))

    try:
        p.run()
    except Exception as ex:
        print(ex)
        pass


def parseArgs():
    parser = argparse.ArgumentParser(description='Poly Counting')
    parser.add_argument(
        '--model',
        dest='model',
        help='model model file',
        default="/opt/poco_model.hdf5",
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
