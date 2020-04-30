import os
from ..core.paras import PreParas, KerasParas
from ..core.dice import dice_coef, dice_coef_loss
from ..core.utils import min_max_normalization, resample_img
from ..core.eval import out_LabelHot_map_2D
from keras.models import load_model
from .. import __version__
import SimpleITK as sitk
import numpy as np
import argparse


def brain_seg_prediction(input_path, output_path,
                         pre_paras, keras_paras):
    # load model
    seg_net = load_model(keras_paras.model_path,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})

    imgobj = sitk.ReadImage(input_path)

    # re-sample to 0.1x0.1x0.1
    resampled_imgobj = resample_img(imgobj,
                                    new_spacing=[0.1, 0.1, 0.1],
                                    interpolator=sitk.sitkLinear)

    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    normed_array = min_max_normalization(img_array)
    out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array,
                                                            seg_net,
                                                            pre_paras,
                                                            keras_paras)

    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_label_img.CopyInformation(resampled_imgobj)

    resampled_label_map = resample_img(out_label_img,
                                       new_spacing=imgobj.GetSpacing(),
                                       new_size=imgobj.GetSize(),
                                       interpolator=sitk.sitkNearestNeighbor)
    # Save the results
    sitk.WriteImage(resampled_label_map, output_path)


def main():
    parser = argparse.ArgumentParser(prog='rbm',
                                     description="Command line tool for Rat Brain Masking with 2D Unet.")
    parser.add_argument("-v", "--version", action='version', version='%(prog)s v{}'.format(__version__))
    parser.add_argument("input", help="The NifTi1 file of rat brain MRI (T2w or EPI)", type=str)
    parser.add_argument("output", help="The destination for brain mask", type=str)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Default Parameters Preparation
    pre_paras = PreParas()
    pre_paras.patch_dims = [1, 128, 128]
    pre_paras.patch_label_dims = [1, 128, 128]
    pre_paras.patch_strides = [1, 32, 32]
    pre_paras.n_class = 2

    # Parameters for Keras model
    keras_paras = KerasParas()
    keras_paras.outID = 0
    keras_paras.thd = 0.5
    keras_paras.loss = 'dice_coef_loss'
    keras_paras.img_format = 'channels_last'
    keras_paras.model_path = os.path.join(os.path.dirname(__file__), 'rat_brain-2d_unet.hdf5')

    brain_seg_prediction(input_path, output_path, pre_paras, keras_paras)


if __name__ == '__main__':
    main()
