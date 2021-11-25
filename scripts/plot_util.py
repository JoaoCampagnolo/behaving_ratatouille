from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

import headband_setup


# this is important else throws error

def plot_drosophila_heatmap(image=None, hm=None, concat=False, draw_joints=None, scale=1):
    '''
    concat: Whether to return a single image or njoints images concatenated
    '''
    assert (image is not None and hm is not None)
    inp = image
    if hm.ndim == 3 and not concat:
        # then sum the joint heatmaps
        if draw_joints is not None:
            hm = hm[draw_joints, :, :]
        hm = hm.sum(axis=0)
    if concat is False:
        img = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2]))
        for i in range(3):
            img[:, :, i] = inp[:, :, i]
        # scale to make it faster
        if scale != 1:
            img = scipy.misc.imresize(img, [int(img.shape[0] / scale), int(img.shape[1] / scale), img.shape[2]])

        hm_resized = scipy.misc.imresize(hm, [img.shape[0], img.shape[1], 3])
        hm_resized = hm_resized.astype(float) / 255

        img = img.copy() * .3
        hm_color = color_heatmap(hm_resized)
        img += hm_color * .7
        return img.astype(np.uint8)
    elif concat:
        concat_list = []
        for idx, hm_ in enumerate(hm):
            if not (idx % 5) in draw_joints or np.max(hm_) == 0.:
                continue
            concat_list.append(plot_drosophila_heatmap(hm_, concat=False))
        return np.hstack(concat_list)


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def color_heatmap(x):
    # x = to_numpy(x)
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def normalize_pose_3d(points3d, normalize_length=False, normalize_median=True):
    if normalize_median:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    if normalize_length:
        length = [0.005, 0.01, 0.01, 0.01, 0.01]
        points3d = points3d.reshape(-1, 15, 3)
        for idx in range(points3d.shape[0]):
            print(idx)
            for j_idx in range(points3d[idx].shape[0]):
                if j_idx % 5 == 4:  # then tarsus-tip
                    continue
                diff = points3d[idx, j_idx + 1, :] - points3d[idx, j_idx, :]
                diff_norm = (diff / np.linalg.norm(diff)) * length[j_idx % 5]
                points3d[idx, j_idx + 1, :] = points3d[idx, j_idx, :] + diff_norm
                next_tarsus_tip = (j_idx - (j_idx % 5)) + 5
                points3d[idx, j_idx + 2:next_tarsus_tip, :] += (diff_norm - diff)
    return points3d
