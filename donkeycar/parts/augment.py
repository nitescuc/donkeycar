'''
    File: augment.py
    Author : Tawn Kramer
    Date : July 2017
'''
import random
from PIL import Image
from PIL import ImageEnhance
import glob
import numpy as np
import math
import cv2

'''
    find_coeffs and persp_transform borrowed from:
    https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
'''
ONE_BY_255 = 1.0 / 255.0


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def rand_persp_transform(img):
    width, height = img.size
    new_width = math.floor(float(width) * random.uniform(0.9, 1.1))
    xshift = math.floor(float(width) * random.uniform(-0.2, 0.2))
    coeffs = find_coeffs(
        [(0, 0), (256, 0), (256, 256), (0, 256)],
        [(0, 0), (256, 0), (new_width, height), (xshift, height)])

    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def augment_image(np_img, shadow_images=None, do_warp_persp=False, apply_blur=False):
    """
    :param np_img: numpy image
        input image in numpy normalised format
    :param shadow_images: list of 2-tuples of PIL images
        shadow vector as prepared by load_shadow_images
    :param do_warp_persp: bool
        apply warping
    :return: numpy image
        output image in numpy normalised format
    """
    # denormalise image to 8int
    conv_img = np_img * 255.0
    conv_img = conv_img.astype(np.uint8)
    # convert to PIL and apply transformation
    img = Image.fromarray(conv_img)
    img = augment_pil_image(img, shadow_images, do_warp_persp)
    if apply_blur:
        img = cv2.bilateralFilter(np.array(img),9,75,75)
    # transform back to normalised numpy format
    img_out = np.array(img).astype(np.float) * ONE_BY_255
    return img_out


def augment_pil_image(img, shadow_images=None, do_warp_persp=False):
    """
    :param img: PIL image
        input image in PIL format
    :param do_warp_persp: bool
        apply warping
    :param shadow_images: list of 2-tuples of PIL images
        shadow vector as prepared by load_shadow_images
    :return: PIL image
        augmented image
    """
    # change the coloration, sharpness, and composite a shadow
    factor = random.uniform(0.5, 2.0)
    img = ImageEnhance.Brightness(img).enhance(factor)
    factor = random.uniform(0.5, 1.0)
    img = ImageEnhance.Contrast(img).enhance(factor)
    factor = random.uniform(0.5, 1.5)
    img = ImageEnhance.Sharpness(img).enhance(factor)
    factor = random.uniform(0.0, 2.0)
    img = ImageEnhance.Color(img).enhance(factor)
    # optionally composite a shadow, prepared from load_shadow_images
    if shadow_images is not None:
        iShad = random.randrange(0, len(shadow_images))
        top, mask = shadow_images[iShad]
        theta = random.randrange(-35, 35)
        mask.rotate(theta)
        top.rotate(theta)
        mask = ImageEnhance.Brightness(mask).enhance(random.uniform(0.3, 1.0))
        offset = (random.randrange(-128, 128), random.randrange(-128, 128))
        img.paste(top, offset, mask)
    # optionally warp perspective
    if do_warp_persp:
        img = rand_persp_transform(img)
    return img


def load_shadow_images(path_mask):
    shadow_images = []
    filenames = glob.glob(path_mask)
    for filename in filenames:
        shadow = Image.open(filename)
        shadow.thumbnail((256, 256))
        channels = shadow.split()
        if len(channels) != 4:
            continue
        r, g, b, a = channels
        top = Image.merge("RGB", (r, g, b))
        mask = Image.merge("L", (a,))
        shadow_images.append((top, mask))
    return shadow_images


