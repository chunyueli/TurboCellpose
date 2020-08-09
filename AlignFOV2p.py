import glob
import os.path
import cv2 as cv
import skimage.io
import numpy as np
import pandas as pd
from pystackreg import StackReg
from skimage import transform as tf
import matplotlib.pyplot as plt
from cellpose import models
from skimage.segmentation import find_boundaries

def AlignFOV2p(path_to_FOV, path_to_masks=[], templateID=0, diameter=None, transformation='affine'):
    """ perform cross-day alignment on field-of-view (FOV) image by Turboreg method.

    The function save the transformation matmatrices and the registered FOV images under the input folder.

    Parameters
    -------------
    path_to_FOV: str
        the path of the folder containing FOV images
    path_to_masks: str (default [ ])
        the path of the folder containing ROI masks. If the value is empty, the code will automatically extract ROI masks using cellpose. If the ROI masks have already obtained, provide the path to the folder can save time.
    templateID: int (default 0)
        choose which FOV image as a template for alignment
    diameter: neuron diameter.
        If the value is None, the diameter will be automatically estimated by Cellpose. Otherwise, the neuron mask will be detected based on the given diameter value by Cellpose.
    transformation: str (default 'affine')
        translation: X and Y translation
        rigid_body: translation + rotation
        scaled_rotation: translation + rotation + scaling
        affine: translation + rotation + scaling + shearing
        bilinear: non-linear transformation; does not preserve straight lines


    Returns
    ----------------
    Tmatrices: list of the trnaformation matrices
    regImages: list of the registered FOV images
    regROIs: list of the registered ROIs masks

    """
    files=get_file_names(path_to_FOV)
    generate_summary(templateID, files)
    imgs=[]

    nimg = len(imgs)
    imgs = [skimage.io.imread(f) for f in files]

    if path_to_masks == []:
        model = models.Cellpose(gpu=False, model_type='cyto')

        if diameter==None:
            masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0])
        else:
            masks, flows, styles, diams = model.eval(imgs, diameter=diameter, channels=[0,0])

        ROIs_mask = generate_ROIs_mask(masks, imgs)
    else:
        ROI_files=get_file_names(path_to_masks)
        ROIs_mask = [skimage.io.imread(f) for f in ROI_files]


    if not (os.path.exists(path_to_FOV+'/ROIs_mask/')):
        os.makedirs(path_to_FOV+'/ROIs_mask/')
    for i in range(len(files)):
        skimage.io.imsave(path_to_FOV+'/ROIs_mask/' + os.path.split(files[i])[-1], ROIs_mask[i])

    Template = imgs[templateID] # FOV_template
    Template = cv.normalize(Template, Template, 0, 255, cv.NORM_MINMAX)
    Template_ROI = ROIs_mask[templateID]

    Tmatrices=[]
    regImages=[]
    regROIs=[]


    print('TurboReg' + ' is running')
    for j in range(len(imgs)):
        if j != templateID:
            print('registering '  + os.path.split(files[j])[-1])
            Regimage = imgs[j]
            Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
            Regimage_ROI = ROIs_mask[j]
            T_matrix, regIm, regROI= Apply_Turboreg_methods(Template, Template_ROI, Regimage, Regimage_ROI, transformation)
            Tmatrices.append(T_matrix)
            regImages.append(regIm)
            regROIs.append(regROI)

    plot_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs)
    return Tmatrices, regImages, regROIs


def get_file_names(folder):
    image_names = []
    image_names.extend(glob.glob(folder + '/*.png'))
    image_names.extend(glob.glob(folder + '/*.jpg'))
    image_names.extend(glob.glob(folder + '/*.jpeg'))
    image_names.extend(glob.glob(folder + '/*.tif'))
    image_names.extend(glob.glob(folder + '/*.tiff'))
    if image_names==[]:
        print('Load image failed: please check the path')
    elif len(image_names)==1:
        print('Error: the folder needs to contain at least two images')
    else:
        return image_names


def generate_summary(ID, files):
    print('Template image:' + os.path.split(files[ID])[-1])
    regfiles=[]
    for j in range(len(files)):
        if j != ID:
            regfiles.append(os.path.split(files[j])[-1])
    print('Registered images:')
    print(regfiles)


def generate_ROIs_mask(masks, imgs):
    ROIs_mask=[]
    nimg = len(imgs)
    for idx in range(nimg):
        raw_mask= np.zeros((imgs[idx].shape[0], imgs[idx].shape[1]), np.uint8)
        maski = masks[idx]
        for n in range(int(maski.max())):
            ipix = (maski==n+1).nonzero()
            if len(ipix[0])>60:
                raw_mask[ipix[0],ipix[1]] = 255
        ROIs_mask.append(raw_mask)
    return ROIs_mask

def Apply_Turboreg_methods(ref_img, ref_mask, mov_img, mov_mask, transf):
    if transf=='affine':
        sr = StackReg(StackReg.AFFINE)
    elif transf=='translation':
        sr = StackReg(StackReg.TRANSLATION)
    elif transf=='rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)
    elif transf=='scaled_rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)
    elif transf=='bilinear':
        sr = StackReg(StackReg.BILINEAR)

    tmats = sr.register(ref_mask, mov_mask)
    tform = tf.AffineTransform(tmats)
    out_rot = tf.warp(mov_img, tform)
    our_reg = cv.normalize(out_rot, out_rot, 0, 255, cv.NORM_MINMAX)
    our_reg = np.array(our_reg, dtype = 'uint8')

    reg_ROI_mask =tf.warp(mov_mask, tform)
    return tmats, our_reg, reg_ROI_mask

def plot_results(path, files, ID, Template, Template_ROI, Tmatrices, regImages, regROIs):
    method='turboreg'
    # save transformation matrix
    if not (os.path.exists(path + '/' + method.upper() + '/')):
        os.makedirs(path + '/' + method.upper() + '/')

    k=0
    for i in range(len(files)):
        if i!=ID:
            raw_data = {'Registered_file': [os.path.split(files[i])[1]],
                        'Template_file': [os.path.split(files[ID])[1]],
                        'Transformation_matrix':[Tmatrices[k]]}
            df = pd.DataFrame(raw_data, columns = ['Registered_file', 'Template_file', 'Transformation_matrix'])
            dfsave=path +'/' + method.upper() + '/' + os.path.split(files[i])[1][:-4] + '.csv'
            df.to_csv(dfsave)

            output_Im=np.zeros([np.size(Template,1), np.size(Template,1), 3], np.uint8)
            outlines1 = np.zeros(Template_ROI.shape, np.bool)
            outlines1[find_boundaries(Template_ROI, mode='inner')] = 1
            outX1, outY1 = np.nonzero(outlines1)
            output_Im[outX1, outY1] = np.array([255, 0, 0])

            outlines2 = np.zeros(regROIs[k].shape, np.bool)
            outlines2[find_boundaries(regROIs[k], mode='inner')] = 1
            outX2, outY2 = np.nonzero(outlines2)
            output_Im[outX2, outY2] = np.array([255, 255, 22])

            img=cv.hconcat([cv.cvtColor(Template, cv.COLOR_GRAY2BGR),cv.cvtColor(regImages[k], cv.COLOR_GRAY2BGR), output_Im])
            skimage.io.imsave(path +'/' + method.upper() + '/results_' + os.path.split(files[i])[1], img)
            k=k+1
