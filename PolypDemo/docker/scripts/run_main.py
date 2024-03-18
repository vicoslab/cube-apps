#!/usr/bin/python3

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from keras import backend as K

import numpy as np
import cv2

import argparse

from utils import rotate_image

cv2.ocl.setUseOpenCL(False)

#RESIZE_RES = (2970, 2100)
RESIZE_RES = (2080, 1470)

DET_THICKNESS = 5
FONT_THICKNESS = 4
FONT_SCALE = 3
FONT_COLOR = (153, 204, 255)

class PModel:

    def __init__(self, modelFile, blockNumber=4, sizeRange=None):

        self.config = tf.compat.v1.ConfigProto()

        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.config.gpu_options.allow_growth = True

        self.blockNumber = blockNumber
        self.sizeRange = sizeRange
        self.modelFile = modelFile

    def __unetModelBlocks(self, blockNumber=4, filterNumber=16):

        inputs = Input((None, None, 3))
        blockFeatures = []

        x = inputs

        for i in range(blockNumber):
            fnCur = filterNumber * (2 ** (i))
            conv1 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
            conv1 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            blockFeatures.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fnCur = filterNumber * (2 ** (blockNumber))
        conv3 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv3 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        x = drop3
        for i in range(blockNumber):
            fnCur = filterNumber * (2 ** (blockNumber - i - 1))
            up8 = Conv2D(fnCur, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([blockFeatures.pop(), up8], axis=3)

            conv8 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            x = conv8

        conv10 = Conv2D(1, 1, activation='sigmoid')(x)
        model = Model(inputs, conv10)

        return inputs, conv10, model

    def __splitAndPredict(self, p_x, p_y, image):

        with tf.compat.v1.Session(config=self.config):

            _, _, self.model = self.__unetModelBlocks(blockNumber=self.blockNumber)
            self.model.load_weights(self.modelFile)

            partsX = p_x * 2 - 1
            partsY = p_y * 2 - 1

            XParts = [((image.shape[1] // p_x) * x // 2, (image.shape[1] // (p_x)) * (x + 2) // 2) for x in range(partsX)]
            YParts = [((image.shape[0] // p_y) * x // 2, (image.shape[0] // (p_y)) * (x + 2) // 2) for x in range(partsY)]
            total_image = np.zeros((image.shape[0], image.shape[1]))

            for y1, y2 in YParts:
                for x1, x2 in XParts:
                    t_img = np.float32(image[y1:y2, x1:x2, :3])
                    in_image = np.expand_dims(t_img, axis=0)

                    print("in_image {}:{} {}:{} -> {}".format(y1, y2, x1, x2, in_image.shape))

                    t_img = self.model.predict(in_image, batch_size=1)
                    total_image[y1:y2, x1:x2] = np.maximum(t_img[0, ..., 0], total_image[y1:y2, x1:x2])

        K.clear_session()

        return total_image

    def __non_max_suppression_reverse(self, msk, filter_size):

        orig_mask = msk.copy()
        kernel = np.ones((filter_size, filter_size), dtype=np.uint8)
        kernel[filter_size // 2, filter_size // 2] = 0
        dilated_mask = cv2.dilate(orig_mask, kernel, iterations=1)
        local_max_mask = (dilated_mask < orig_mask).astype(np.uint8)
        local_plateau_mask = (dilated_mask == orig_mask).astype(np.uint8)
        local_plateau_mask[dilated_mask == 0] = 0

        _, _, _, centroids = cv2.connectedComponentsWithStats(local_plateau_mask)
        centroids = centroids[~np.isnan(centroids).any(axis=1)]
        centroids = centroids.astype(int)
        ret_mask = np.zeros((orig_mask.shape), dtype=orig_mask.dtype)

        ret_mask[centroids[:, 1], centroids[:, 0]] = 1
        ret_mask[local_max_mask > 0] = 1

        return ret_mask

    def __generate_detections_from_mask(self, mask, threshold=127):

        uint_mask = np.array(mask * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(uint_mask, threshold, 255, 0)

        dist_transform = cv2.distanceTransform(thresh.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        skeleton = self.__non_max_suppression_reverse(dist_transform, 11)
        row_ind, col_ind = np.where(skeleton > 0)
        boxes = np.array(
            [[x[1] - dist_transform[x[0], x[1]] * 1, x[0] - dist_transform[x[0], x[1]] * 1, 2 * int(dist_transform[x[0], x[1]]), 2 * int(dist_transform[x[0], x[1]])] for x in
             zip(row_ind, col_ind)])

        return boxes

    def __non_max_sup_boxes(self, boxes, overlapThresh=0.5):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def predict(self, image_full):
        print("Start")
        bw = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
        threshold = 150
        r, t = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 30
        d = cv2.dilate(t, np.ones((kernel_size, kernel_size)))
        e = cv2.erode(d, np.ones((kernel_size, kernel_size)))
        e2 = cv2.erode(e, np.ones((kernel_size, kernel_size)))
        d2 = cv2.dilate(e2, np.ones((kernel_size, kernel_size)))

        ccs = cv2.connectedComponentsWithStats(d2)
        ix = np.argsort(-ccs[2][1:, -1])[0]
        xc, yc, wc, hc, area = ccs[2][ix + 1]
        cc_mask = ccs[1] == ix + 1
        contours, _ = cv2.findContours(image=cc_mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        _, (w_min, h_min), angle = cv2.minAreaRect(contours[0])
        cutout = image_full[yc:yc + hc, xc:xc + wc]
        rotated = rotate_image(cutout, angle)
        h_r, w_r, _ = rotated.shape
        y_rc, x_rc = h_r // 2, w_r // 2
        final_crop = 120
        final_cropped = rotated[int(y_rc - h_min // 2) + final_crop:int(y_rc + h_min // 2) - final_crop, int(x_rc - w_min // 2) + final_crop:int(x_rc + w_min // 2) - final_crop]
        if final_cropped.shape[0] > final_cropped.shape[1]:
            final_cropped = cv2.rotate(final_cropped, cv2.ROTATE_90_CLOCKWISE)

        resized_input = cv2.resize(final_cropped, RESIZE_RES)

        draw_det_image = np.zeros(resized_input.shape[:2], dtype=np.uint8)

        print(f"Working with size of resized_input:{resized_input.shape}")

        image = resized_input
        
        D = 512

        xSmall = ySmall = False
        if image.shape[0] < D:
            print("Too small x axis")
            xSmall = True
        if image.shape[1] < D:
            print("Too small y axis")
            ySmall = True

        image = cv2.resize(image, (D if ySmall else image.shape[1], D if xSmall else image.shape[0]), interpolation=cv2.INTER_AREA) if any([xSmall, ySmall]) else image

        print("Prediction request: image shape {}".format(image.shape))

        origImage = image[..., ::-1].astype(np.float32) / 255.0
        origShape = origImage.shape
        refShape = origImage.shape

        partsY = refShape[0] // D
        partsX = refShape[1] // D
        divisorPower = 1

        print("PartsY {} PartsX {}".format(partsY, partsX))

        while partsY % (2 ** divisorPower) == 0 and partsX % (2 ** divisorPower) == 0:
            divisorPower += 1
        divisorRequired = (2 ** (self.blockNumber - divisorPower + 2))

        print("DivisorRequired {}".format(divisorRequired))

        refShape = [refShape[0] // (partsY * divisorRequired) * (partsY * divisorRequired),
                    refShape[1] // (partsX * divisorRequired) * (partsX * divisorRequired),
                    refShape[2]]

        print("RefShape {}".format(refShape))

        origImage = cv2.resize(origImage, (refShape[1], refShape[0]), interpolation=cv2.INTER_AREA)
        
        print("OrigImage shape {}".format(origImage.shape))
    
        maskResult = self.__splitAndPredict(partsX, partsY, origImage)
        maskResult = cv2.resize(maskResult, (maskResult.shape[1] // 2, maskResult.shape[0] // 2), interpolation=cv2.INTER_AREA)
        maskResult = maskResult / np.max(maskResult)
        
        tBoxes = self.__generate_detections_from_mask(maskResult, threshold=50)

        print("Found {} boxes".format(len(tBoxes)))

        if (len(tBoxes) != 0):

            tBoxes[:, 0] = tBoxes[:, 0] / maskResult.shape[0] * origShape[0]
            tBoxes[:, 1] = tBoxes[:, 1] / maskResult.shape[1] * origShape[1]
            tBoxes[:, 2] = tBoxes[:, 2] / maskResult.shape[0] * origShape[0]
            tBoxes[:, 3] = tBoxes[:, 3] / maskResult.shape[1] * origShape[1]

            tBoxes = np.array(list(filter(lambda x: (x[2] * x[3] > 0) and (x[0] > 0) and (x[1] > 0), tBoxes)))
            if self.sizeRange != None:
                tBoxes = np.array(list(filter(lambda x: x[2] < 1.0 * self.sizeRange[1] and x[3] < 1.0 * self.sizeRange[1] and
                                                        x[2] >= 0.5 * self.sizeRange[0] and x[3] >= 0.5 * self.sizeRange[0] and
                                                        x[0] > 0.0 and x[1] > 0.0, tBoxes)))

            tBoxes[:, 0] = tBoxes[:, 0] + tBoxes[:, 2] // 2
            tBoxes[:, 1] = tBoxes[:, 1] + tBoxes[:, 3] // 2

            tBoxes = self.__non_max_sup_boxes(tBoxes, overlapThresh=0.7)

            for b in tBoxes:
                xd = b[0]
                yd = b[1]
                wd = b[2]
                hd = b[3]
                
                # Skip detection that will be outside the bounds of the image
                if (xd - wd // 2 >= 0) and (yd - hd // 2 >= 0) and (xd + wd // 2 < origShape[1]) and (yd + hd // 2 < origShape[0]):
                    cv2.circle(draw_det_image, (xd, yd), radius=wd//2, color=255, thickness=DET_THICKNESS)

        draw_det_image = cv2.resize(draw_det_image, dsize=(final_cropped.shape[1], final_cropped.shape[0]))
        
        mask = np.zeros((rotated.shape[0], rotated.shape[1]), dtype=np.uint8)
        mask[int(y_rc - h_min // 2) + final_crop:int(y_rc + h_min // 2) - final_crop, int(x_rc - w_min // 2) + final_crop:int(x_rc + w_min // 2) - final_crop] = draw_det_image
        
        mask = rotate_image(mask, - angle)

        t = int(mask.shape[0]//2 - cutout.shape[0]// 2)
        b = int(t + cutout.shape[0])
        l = int(mask.shape[1]//2 - cutout.shape[1]// 2)
        r = int(l + cutout.shape[1])

        # Check if mask is not large enough
        if t < 0 or b > mask.shape[0] or l < 0 or r > mask.shape[1]:
            # Calculate padding values
            top_pad = abs(t) if t < 0 else 0
            bottom_pad = b - mask.shape[0] if b > mask.shape[0] else 0
            left_pad = abs(l) if l < 0 else 0
            right_pad = r - mask.shape[1] if r > mask.shape[1] else 0

            # Pad the mask
            mask = cv2.copyMakeBorder(mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

            # Update t, b, l, r after padding
            t += top_pad
            b += bottom_pad
            l += left_pad
            r += right_pad

        mask = mask[t:b, l:r]

        mask_full = np.zeros(image_full.shape[:2], dtype=np.uint8)
        mask_full[yc:yc + hc, xc:xc + wc] = mask
        mask_full = (mask_full > 128).astype(np.uint8)
        
        mask_full3 = np.stack((mask_full, mask_full, mask_full), axis=-1)
        green = np.stack((np.zeros_like(mask_full), np.ones_like(mask_full) * 255, np.zeros_like(mask_full)), axis=-1)

        final = mask_full3 * green + (1 - mask_full3) * image_full

        cv2.putText(final, "Stevilo detektiranih polipov je {}".format(len(tBoxes)), (300, image_full.shape[0]-300), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=FONT_SCALE, color=FONT_COLOR, thickness=FONT_THICKNESS)

        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)


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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                frame = self.detection_method.predict(frame)
            except Exception as e:
                print(e)
                cv2.putText(frame, f"Failed to detect image", (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, [0, 0, 255], 10)

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
        default="/opt/model.hdf5",
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
