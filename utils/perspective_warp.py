import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def random_perspective_matrix_gen(origin_shape, destination_shape):
    # x0 y0 move image center to (0,0)
    x0 = -origin_shape[1]/2.
    y0 = -origin_shape[0]/2.
    x1 = destination_shape[1]/2.
    y1 = destination_shape[0]/2.
    # Prepare the transformation matrix
    # Get Euler angles
    ang_z = (np.random.random_sample() - 0.5) * 10.
    ang_y = (np.random.random_sample() - 0.5) * 60.
    ang_x = (np.random.random_sample() - 0.5) * 30.
    rot_m = R.from_euler('zyx', [ang_z, ang_y, ang_x], degrees=True).as_matrix()
    # Move the display in the room (Translation)
    del_x = (np.random.random_sample() - 0.5) * 500.
    del_y = (np.random.random_sample() - 0.5) * 500.
    del_z = np.random.random_sample() * (-1500.) - 3000.
    # Set the viewbox constants (http://www.songho.ca/opengl/gl_projectionmatrix.html)
    t = 10.
    r = 10.
    n = 8000.
    f = 1500000.
    # Perspective transformation matrix
    return np.array([
        [n/r*rot_m[0,0] - x1*rot_m[2,0], n/r*rot_m[0,1] - x1*rot_m[2,1], n/r*(rot_m[0,0]*x0 + rot_m[0,1]*y0 + del_x) - x1*(rot_m[2,0]*x0 + rot_m[2,1]*y0 + del_z)],
        [n/t*rot_m[1,0] - y1*rot_m[2,0], n/t*rot_m[1,1] - y1*rot_m[2,1], n/t*(rot_m[1,0]*x0 + rot_m[1,1]*y0 + del_y) - y1*(rot_m[2,0]*x0 + rot_m[2,1]*y0 + del_z)],
        [-rot_m[2,0], -rot_m[2,1], -(rot_m[2,0]*x0 + rot_m[2,1]*y0 + del_z)],
    ])

def warp_and_add(img_f, img_b, trans_m):
    # Warp the forground image
    interpo_flag = cv2.INTER_LINEAR
    img_mask = np.ones(img_f.shape[:2]) * 255
    img_w = cv2.warpPerspective(img_f, trans_m, img_b.shape[1::-1], flags=interpo_flag)
    img_mask_w = cv2.warpPerspective(img_mask, trans_m, img_b.shape[1::-1], flags=interpo_flag)[:, :, np.newaxis]
    # print("mask max = {}".format(np.amax(img_mask_w)))
    # print("mask min = {}".format(np.amin(img_mask_w)))
    # print("img_w datatype is {}".format(img_w.dtype))
    foreground = np.uint8(img_w * (img_mask_w / 255.))
    background = np.uint8(img_b * (1. - img_mask_w / 255.))
    return cv2.add(foreground, background)
    # dst = cv2.GaussianBlur(dst, (0, 0), 2)