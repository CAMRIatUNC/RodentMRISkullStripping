import numpy as np
import SimpleITK as sitk


def min_max_normalization(img):
    new_img = img.copy()
    new_img = new_img.astype(np.float32)

    min_val = np.min(new_img)
    max_val = np.max(new_img)
    new_img =(np.asarray(new_img).astype(np.float32) - min_val)/(max_val-min_val)
    return new_img


def dim_2_categorical(label, num_class):
    dims = label.ndim
    if dims == 2:
        col, row = label.shape
        ex_label = np.zeros((num_class, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    elif dims == 3:
        leng,col,row = label.shape
        ex_label = np.zeros((num_class, leng, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    else:
        raise Exception
    return ex_label


def resample_img(imgobj, new_spacing, interpolator, new_size=None):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(imgobj.GetDirection())
    resample.SetOutputOrigin(imgobj.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    if new_size is None:
        orig_size = np.array(imgobj.GetSize(), dtype=np.int)
        orig_spacing = np.array(imgobj.GetSpacing())
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]

    resample.SetSize(new_size)

    resampled_imgobj = resample.Execute(imgobj)
    return resampled_imgobj