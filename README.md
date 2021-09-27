% Rodent Brain-mask Segmentation tool


Detail description: 
The ‘RodentMRISkullStripping’ is a python module designed to provide an automatic rodent skull stripping. This module is a deep-learning-based framework, U-Net, to automatically identify the rodent brain boundaries in MR images. The whole framework is implemented using Keras with TensorFlow as the backend.


Plese install the following package before using RodentMRISkullStripping:
1) numpy, tensorflow, and keras
ex: conda install numpy tensorflow keras

2) SimpleITK
ex: conda install -c https://conda.anaconda.org/simpleitk SimpleITK

3) scikit-image
ex: conda install -c conda-forge scikit-image


Instruction:
1) install RodentMRISKullStripping
'pip install rbm'
2) Give the input and output folder, you should put all the image in NIfTI format in the input folder .
'rbm -s 0.01 <input> <output>'
-s: resample resolution (0.05 or 0.01)
