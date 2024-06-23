import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sewar.full_ref as full_ref
import numpy as np
import cv2
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def compute_ergas(img1, img2, scale):
    d = img1 - img2
    ergasroot = 0
    for i in range(d.shape[2]):
        ergasroot = ergasroot + np.mean(d[:, :, i] ** 2) / np.mean(img1[:, :, i])**2

    ergas = 100 / scale * np.sqrt(ergasroot / d.shape[2])
    return ergas / 1.5


def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim ==3

    img_c, img_w, img_h = img1.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar)**2, 1)
    max1 = 1#np.max(ref, 1)

    psnrall = 10*np.log10(max1**2/msr)
    out_mean = np.mean(psnrall)
    return out_mean


def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 1e-4
    x_true[np.where((np.linalg.norm(x_true, 2, 1)) == 0),] += 1e-4

    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    sam = np.clip(sam, 0, 1)

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()

    return mSAM

def metrics(GT, P, scale):  # c,w,h
    psnr = compute_psnr(GT, P)
    GT = GT.transpose(1, 2, 0)
    P = P.transpose(1, 2, 0)
    sam = compute_sam(GT, P)  # sam
    ergas = compute_ergas(GT, P, scale)

    from skimage.measure import compare_ssim as ssim
    ssims = []
    for i in range(GT.shape[2]):
        ssimi = ssim(GT[:,:,i], P[:,:,i], data_range=P[:,:,i].max() - P[:,:,i].min())
        ssims.append(ssimi)
    ssim = np.mean(ssims)

    uqi = full_ref.uqi(GT, P)

    return np.float64(psnr), np.float64(sam), ergas, ssim, uqi


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range

    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    import math
    shave = math.ceil(shave)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, scale=2, benchmark=False):
    if benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
