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


def crop_part(image, final_crop):
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

    ccs = cv2.connectedComponentsWithStats(dilated)

    i = 1
    x, y, w, h, area = ccs[2][i]
    cc_mask = ccs[1] == i
    contours, _ = cv2.findContours(image=cc_mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    _, (w_min, h_min), angle = cv2.minAreaRect(contours[0])

    points_bb_rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    bb_rect = cv2.fillPoly(np.zeros_like(bw), [points_bb_rect], (255))
    cutout = image[y:y + h, x:x + w]
    rotated = rotate_image(cutout, angle)

    h_r, w_r, _ = rotated.shape
    y_rc, x_rc = h_r // 2, w_r // 2
    final = rotated[int(y_rc - h_min // 2):int(y_rc + h_min // 2), int(x_rc - w_min // 2):int(x_rc + w_min // 2)]
    final_cropped = final[final_crop:-final_crop, final_crop:-final_crop]

    if final_cropped.shape[0] < final.shape[1]:
        final_cropped = cv2.rotate(final_cropped, cv2.ROTATE_90_CLOCKWISE)

    ty = int(y + h // 2) + 45
    tx = int(x + w // 2) - 45

    return final_cropped, contours, tx, ty
