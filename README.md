# Squeeze and Excitation Blocks for Fully ConvNets

PyTorch Implementation of 'squeeze and excitation' blocks for Fully Convolutional Neural Networks

Authors: Abhijit Guha Roy (https://github.com/abhi4ssj), Shayan Siddiqui (https://github.com/shayansiddiqui) and Anne-Marie Rickmann (https://github.com/arickm)

Manuscipt for details: https://arxiv.org/abs/1808.08127, https://arxiv.org/abs/1906.04649

------------------------

New Additions

(i) 3D version of Spatial Squeeze and Channel Excitation (cSE) Block

(ii) 3D version of Channel Squeeze and Spatial Excitation (sSE) Block

(iii) 3D version of Concurrent Spatial and Channel 'Squeeze and Excitation' (scSE) Block

(iv) 3D Project and Excite Block (Link: https://arxiv.org/abs/1906.04649)

For using these 3D extensions, Please cite

```
@inproceedings{rickmann2019project,
  title={`Project \& Excite' Modules for Segmentation of Volumetric Medical Scans},
  author={Rickmann, Anne-Marie and Sarasua, Ignacio and Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2019},
  organization={Springer}
```

Implementation includes 

(i) Spatial Squeeze and Channel Excitation (cSE) Block

Reference:

Hu, J., Shen, L. and Sun, G., 2018. Squeeze-and-excitation networks. In Proc. CVPR.

Link: https://arxiv.org/abs/1709.01507

(ii) Channel Squeeze and Spatial Excitation (sSE) Block

(iii) Concurrent Spatial and Channel 'Squeeze and Excitation' (scSE) Block

### Please cite:
```
@inproceedings{roy2018concurrent,
  title={Concurrent Spatial and Channel ‘Squeeze \& Excitation’in Fully Convolutional Networks},
  author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={421--429},
  year={2018},
  organization={Springer}
}
```
Link: https://arxiv.org/abs/1803.02579
```
@article{roy2019recalibrating,
  title={Recalibrating Fully Convolutional Networks With Spatial and Channel “Squeeze and Excitation” Blocks},
  author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={2},
  pages={540--549},
  year={2019},
  publisher={IEEE}
}
```
Link: https://arxiv.org/abs/1808.08127


## Pre-requisites

You need to have following in order for this library to work as expected
1. Python >= 3.5
2. Pytorch >= 1.0.0
3. Numpy >= 1.14.0

## Installation

Always use the latest release. Use following command with appropriate version no(v1.0) in this particular case to install. You can find the link for the latest release in the release section of this github repo

```
pip install https://github.com/abhi4ssj/squeeze_and_excitation/releases/download/v1.0/squeeze_and_excitation-1.0-py2.py3-none-any.whl
```

## How to Use

Please use the following link to read the technical documentation

https://abhi4ssj.github.io/squeeze_and_excitation/


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)
