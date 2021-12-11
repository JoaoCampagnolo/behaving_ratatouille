# File:             frame_preprocess.py
# Date:             Winter 2021
# Description:      Preprocesses the raw videos from behavioral experiments (TODO: better description :d)
#                   The goal is to define an ROI around the animal, proceed to mask the frames and align the orientation
#                   of its body. The intent of this pipeline is to yield videos that are fit for posture extraction by means
#                   of PCA.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
# from skimage.registration import phase_cross_correlation
# from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
import sys
import math
import scipy
import scipy.ndimage.interpolation as ndii
from skimage.feature import register_translation
import pprint
import time
import h5py


def process_videos(videos, trackings, root_dir, template_frms=None, temp_frm=320, crop_h=100, crop_w=100, canny_t1=300, canny_t2=220, 
                   morph_iter=7, background=255, min_frm=0, max_frm=None):
    '''
    Preprocesses the raw videos from behavioral experiments (TODO: better description :d)
    The goal is to define an ROI around the animal, proceed to mask the frames and align the orientation
    of its body. The intent of this pipeline is to yield videos that are fit for posture extraction by means
    of PCA.
    TODO: set videos and trackings in the Behav_annotation class when creating training and validation sets.
    '''
    # Exctract a list of full video paths:
    vid_paths = [find_path(vid,root_dir)[-1] for vid in videos]
    
    # Exctract a list of full tracking paths:
    track_paths = [find_path(tck,root_dir)[-1] for tck in trackings]
    assert len(vid_paths) == len(track_paths), 'Number of videos and h5 files do not match'
    # TODO
    # Maybe assert if names match
    # Check if multiple files with the same name exist
    
    # Set template frames to default if they are not provided
    if template_frms is None:
        template_frms = [temp_frm]*len(vid_paths)
        
    # Output video paths
    out_pahts = []
    
    # Iterate over list of video paths:
    for video, dlc_train, template in zip(vid_paths, track_paths, template_frms):
        vid_id = vid_paths.index(video)
        vidcap = cv2.VideoCapture(video)
        n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        count = 0
        
        # Create a nested directory for each video:
        vid_tag = '_'.join(os.path.basename(video).split('_')[1:4])
        if not os.path.exists(os.path.join(root_dir, f'{vid_tag}_frames')):
            os.makedirs(os.path.join(root_dir, f'{vid_tag}_frames'))
            print(f'Creating directory: {os.path.join(root_dir, f"{vid_tag}_frames")}')
        
            # Iterate over every frame: 
            # (this is inside the loop, so if this dir exists, it won't write the frames again)
            print(f'Writing {vid_tag} video frames')
            while success:
                # Write the frame - maybe this is avoidable, I just want to keep the frames to assess possible errors
                cv2.imwrite(os.path.join(os.path.join(root_dir, f'{vid_tag}_frames'), f'frame_{count}.jpg'), image)   
                success,image = vidcap.read()
                if count%5000 == 0 and count > 0:
                    print(f'{count} frames written')
                count += 1
            print(f'Extracted {count} frames')
        
        # Load DLC's data:
        print('Loading pose from DLC tracking algorithm')
        h5 = h5py.File(dlc_train,'r')
        # Check h5 file if needed:
#         def printname(name):
#             print(name)
#         h5.visit(printname)
        indices = h5['df_with_missing/table']
        (dlc_frms, x, y, conf) = np.transpose(np.asarray(
            [(entry[0], entry[1][0], entry[1][1], entry[1][2]) for entry in indices]))
        dlc_frms = [int(frame) for frame in dlc_frms]
        # TODO - Check if there are NaNs in the pose file and skip or deal with those values
        # Also smooth the coordinates with a 1€ filter, in case DLC misses some frames
        assert n_frames == len(dlc_frms), f'Length of frames in video and DLC model do not match: {len(count)}=/={len(dlc_frms)}'
        
        # Set the template frame
        print(f'Setting up template frame #{template} for rotational, translational and scaling corrections')
        img = cv2.imread(os.path.join(os.path.join(root_dir, f'{vid_tag}_frames'), f'frame_{template}.jpg'), 0)
        # Crop the frame
        template_img = crop_frames(img, x[template], y[template], fh=crop_h, fw=crop_w)
        # Apply Canny Edge filter and morphologically open and close the edges
        closed_img = close_edges(template_img, th1=canny_t1, th2=canny_t2, close_iter=morph_iter)
        # Apply the mask over the frame
        template_img[closed_img==0] = background
        
        # Set output video parameters:
        print(f'Creating video from {vid_tag}')
        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(os.path.join(os.path.join(root_dir, f'{vid_tag}_frames'), f'{vid_id}_pose.avi'),
                                fourcc, 1, (crop_w, crop_h))
            
        # Reload the frames and get pose from DLC's h5 file
        if max_frm is None:
            max_frm = n_frames
        for frame in range(min_frm, max_frm):
            if frame%5000 == 0 and frame > 0:
                    print(f'{frame} frames aligned')
            # Reload the frame
            img = cv2.imread(os.path.join(os.path.join(root_dir, f'{vid_tag}_frames'), f'frame_{frame}.jpg'), 0)
            # Crop the frame
            crop_img = crop_frames(img, x[frame], y[frame], fh=crop_h, fw=crop_w)
            # Apply Canny Edge filter and morphologically open and close the edges
            closed_img = close_edges(crop_img, th1=canny_t1, th2=canny_t2, close_iter=morph_iter)
            # Apply the mask over the frame
            crop_img[closed_img==0] = background 
            # Asses the rotational translation between the current frame and the reference frame
            angle, rotated1, rotated2 = rotational_shift(template_img, crop_img)
            rot_img = rotate_image(crop_img, angle)
            # Crop the edges of the rotated image
            curr_frm = crop_edges(rot_img, crop_w, crop_h, angle)
            # Translationally align the frame with the template frame
            shift, error, diffphase = register_translation(curr_frm, template_img)
            image_product = np.fft.fft2(template_img) * np.fft.fft2(curr_frm).conj()
            cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
            final_img = scipy.ndimage.shift(curr_frm, -shift, mode='constant', cval=255)
            # Write video
            video.write(final_img)
            
        # Close video
        out_paths.append(os.path.join(os.path.join(root_dir, f'{vid_tag}_frames'), f'{vid_id}_pose.avi'))
        cv2.destroyAllWindows()
        video.release()
        
    return out_paths


def find_path(file, root_dir):
    path = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name == file:
                path.append(os.path.abspath(os.path.join(root, name)))
    return path


def close_edges(img, th1=260, th2=150, kx=3, ky=3, dil_iter=1, close_iter=6):
    edges = cv2.Canny(img,th1,th2)
    kernel = np.ones((kx,ky), np.uint8)
    # img_erosion = cv2.erode(edges, kernel, iterations=1)
    img_dilation = cv2.dilate(edges, kernel, iterations=dil_iter)
    closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    closing = (closing > 0).astype(np.uint8)
    return closing


def rotational_shift(image1, image2):
    def compute_angle(image):
        # Convert to grayscale, invert, and Otsu's threshold
        gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find coordinates of all pixel values greater than zero
        # then compute minimum rotated bounding box of all coordinates
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # The cv2.minAreaRect() function returns values in the range
        # [-90, 0) so need to correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image to horizontal position 
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                  borderMode=cv2.BORDER_REPLICATE)

        return (angle, rotated)

    angle1, rotated1 = compute_angle(image1)
    angle2, rotated2 = compute_angle(image2)
        
    # Both angles are positive
    if angle1 >= 0 and angle2 >= 0:
        difference_angle = abs(angle1 - angle2)
    # One positive, one negative
    elif (angle1 <= 0 and angle2 > 0) or (angle1 >= 0 and angle2 < 0) or (angle1 < 0 and angle2 >= 0) or (angle1 < 0 and angle2 >= 0):
        difference_angle = abs(angle1) + abs(angle2)
    # Both negative
    elif angle1 < 0 and angle2 < 0:
        angle1 = abs(angle1)
        angle2 = abs(angle2)
        difference_angle = max(angle1, angle2) - min(angle1, angle2)

    return (difference_angle, rotated1, rotated2)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# From https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr


def crop_edges(img, w, h, angle, mult=.90, pix_val=255):
    copy = img.copy()
    bw, bh = rotatedRectWithMaxArea(w, h, angle)
    bw *= mult
    bh *= mult
    pmask = [[i<(w-bw)//2 or i>(w+bw)//2 or j<(h-bh)//2 or j>(h+bh)//2 
              for i in range(np.shape(rot_img)[0])] for j in range(np.shape(rot_img)[1])]
    copy[pmask] = pix_val
    return copy


