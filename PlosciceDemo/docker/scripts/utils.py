import pickle
import cv2
import numpy as np

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


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


def pickle_load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def crop_part_single(image, final_crop):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 50
    tried = []
    while True:
        tried.append(threshold)
        r, t = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 25
        eroded = cv2.erode(t, np.ones((kernel_size, kernel_size)))
        dilated = cv2.dilate(eroded, np.ones((kernel_size, kernel_size)))
        ccs = cv2.connectedComponentsWithStats(dilated)
        if ccs[0] > 2:
            threshold += 5
        elif ccs[0] < 2:
            threshold -= 1
        else:
            break
        if threshold in tried:
            print(f"Cant find suitable threshold!")
            return None, None, 0, 0

    ccs = cv2.connectedComponentsWithStats(dilated)

    i = 1
    x, y, w, h, area = ccs[2][i]
    cc_mask = ccs[1] == i
    contours, _ = cv2.findContours(image=cc_mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    _, (w_min, h_min), angle = cv2.minAreaRect(contours[0])

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

    if final_cropped.shape[0] < final_cropped.shape[1]:
        final_cropped = cv2.rotate(final_cropped, cv2.ROTATE_90_CLOCKWISE)

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

