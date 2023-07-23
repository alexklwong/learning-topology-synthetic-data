# Learning Topology from Synthetic Data for Unsupervised Depth Completion

PyTorch implementation of *Learning Topology from Synthetic Data for Unsupervised Depth Completion*

Published in RA-L January 2021 and ICRA 2021

[[publication]](https://ieeexplore.ieee.org/document/9351588) [[arxiv]](https://arxiv.org/pdf/2106.02994v1.pdf) [[talk]](https://www.youtube.com/watch?v=zGKH-OKPJD4)

Model have been tested on Ubuntu 20.04 using Python 3.7, 3.8, PyTorch 1.10 and CUDA 11.1

Authors: [Alex Wong](http://web.cs.ucla.edu/~alexw/), [Safa Cicek](https://bsafacicek.github.io/)

If this work is useful to you, please cite our paper:
```
@article{wong2021learning,
    title={Learning topology from synthetic data for unsupervised depth completion},
    author={Wong, Alex and Cicek, Safa and Soatto, Stefano},
    journal={IEEE Robotics and Automation Letters},
    volume={6},
    number={2},
    pages={1495--1502},
    year={2021},
    publisher={IEEE}
}
```

### Looking our latest work in unsupervised depth completion?

Checkout our ICCV 2021 oral paper, [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*

[KBNet][kbnet_github] runs at 15 ms/frame (67 fps) and improves over ScaffNet on both indoor (VOID) and outdoor (KITTI) performance!

We have just released the PyTorch version of [VOICED](voiced_github): *Unsupervised Depth Completion from Visual Inertial Odometry*!

**Table of Contents**
1. [Setting up](#setting-up)
2. [Downloading pretrained models](#downloading-pretrained-models)
3. [Running ScaffNet and FusionNet](#running-scaffnet-fusionnet)
4. [Training ScaffNet and FusionNet](#training-scaffnet-fusionnet)
5. [Related projects](#related-projects)
6. [License and disclaimer](#license-disclaimer)

For all setup, training and evaluation code below, we assume that your current working directory is in
```
/path/to/learning-topology-synthetic-data/tensorflow
```

to check that this is the case, you can use `pwd`.
```
pwd
```

## Setting up your virtual environment <a name="setting-up"></a>
We will create a virtual environment with the necessary dependencies
```
virtualenv -p /usr/bin/python3.8 voiced-torch-py3env
source voiced-torch-py3env/bin/activate
pip install -r requirements.txt
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Setting up your datasets
For datasets, we will use [Virtual KITTI 2][vkitti_dataset] and [KITTI][kitti_dataset] for outdoors and [SceneNet][scenenet_dataset] and [VOID][void_github] for indoors. Note: Unlike the original Tensorflow implementation, we use Virtual KITTI 2 and the setup script operates differently. We assume you have extracted the `vkitti_2.0.3_depth` within `virtual_kitti` directory i.e. `/path/to/virtual_kitti/vkitti_1.3.1_depthgt`. If you already have all the datasets downloaded and extracted to the right form, you can create symbolic links to them via
```
mkdir data
ln -s /path/to/virtual_kitti data/
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_depth_completion data/
ln -s /path/to/scenenet data/
ln -s /path/to/void_release data/
```

In case you do not already have KITTI, Virtual KITTI 2, and VOID datasets downloaded, we provide download scripts for them:
```
bash bash/setup_dataset_kitti.sh
bash bash/setup_dataset_void.sh
bash bash/setup_dataset_vkitti.sh
```

The `bash/setup_dataset_void.sh` script downloads the VOID dataset using gdown. However, gdown intermittently fails. As a workaround, you may download them via:
```
https://drive.google.com/open?id=1kZ6ALxCzhQP8Tq1enMyNhjclVNzG8ODA
https://drive.google.com/open?id=1ys5EwYK6i8yvLcln6Av6GwxOhMGb068m
https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI
```
which will give you three files `void_150.zip`, `void_500.zip`, `void_1500.zip`.

Assuming you are in the root of the repository, to construct the same dataset structure as the setup script above:
```
mkdir void_release
unzip -o void_150.zip -d void_release/
unzip -o void_500.zip -d void_release/
unzip -o void_1500.zip -d void_release/
bash bash/setup_dataset_void.sh unpack-only
```
If you encounter `error: invalid zip file with overlapped components (possible zip bomb)`. Please do the following
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
```
and run the above again.

For more detailed instructions on downloading and using VOID and obtaining the raw rosbags, you may visit the [VOID][void_github] dataset webpage.

## Downloading our pretrained models <a name="downloading-pretrained-models"></a>
To use our ScaffNet models trained on Virtual KITTI and SceneNet datasets, and our FusionNet models trained on KITTI and VOID datasets, you can download them from Google Drive
```
gdown https://drive.google.com/uc?id=1PSDGD7k8uf_IwAqQnXiHBBQyG_wn_hmT
unzip pretrained_models-pytorch.zip
```

Note: `gdown` fails intermittently and complains about permission. If that happens, you may also download the models via:
```
https://drive.google.com/file/d/1PSDGD7k8uf_IwAqQnXiHBBQyG_wn_hmT/view?usp=sharing
```

We note that if you would like to directly [train FusionNet](#training-scaffnet-fusionnet), you may use our pretrained ScaffNet model.

For reproducibility, we have retrained both ScaffNet and FusionNet using the PyTorch implementation. We have released the pretrained weights in the `pretrained_models` directory. For example
```
pretrained_models/scaffnet/scenenet/scaffnet-scenenet.pth
pretrained_models/scaffnet/vkitti/scaffnet-vkitti.pth
pretrained_models/fusionnet/void/fusionnet-void1500.pth
```
Note that our PyTorch version of ScaffNet and FusionNet models on VOID1500 improves on MAE and RMSE over the Tensorflow version, but performs worse on iMAE and iRMSE. In terms of architecture, much of it stays the same, but we have modified the kernel size of spatial pyramid pooling to be 13, 17, 19, 21, and 25 to support both indoor and outdoor with the same set of parameters. We will be releasing more pretrained models i.e. FusionNet on KITTI over the upcoming months. Stay tuned!

For KITTI:
| Model                              | MAE    | RMSE    | iMAE  | iRMSE |
| :----------------------------------| :----: | :-----: | :---: | :---: |
| ScaffNet (paper - Tensorflow)      | 318.42 | 1425.54 | 1.40  | 5.01  |
| ScaffNet (retrained - Tensorflow)  | 317.17 | 1425.95 | 1.40  | 4.95  |
| ScaffNet (retrained - PyTorch)     | 346.67 | 1390.27 | 1.56  | 5.04 |
| FusionNet (paper - Tensorflow)     | 286.32 | 1182.78 | 1.18  | 3.55  |
| FusionNet (retrained - Tensorflow) | 282.97 | 1184.36 | 1.17  | 3.48  |
| FusionNet (retrained - PyTorch)    |    -   |     -   |   -   |   -   |

For VOID:
| Model                              | MAE    | RMSE    | iMAE  | iRMSE  |
| :--------------------------------- | :----: | :-----: | :---: | :----: |
| ScaffNet (paper - Tensorflow)      | 72.88  | 162.75  | 42.56 | 90.15  |
| ScaffNet (retrained - Tensorflow)  | 65.90  | 153.96  | 35.62 | 77.73  |
| ScaffNet (retrained - PyTorch)     | 61.10  | 130.90  | 45.66 | 134.44 |
| FusionNet (paper - Tensorflow)     | 60.68  | 122.01  | 35.24 | 67.34  |
| FusionNet (retrained - Tensorflow) | 56.24  | 117.94  | 31.58 | 63.78  |
| FusionNet (retrained - PyTorch)    | 54.11  | 116.96  | 34.00 | 68.99  |

## Running ScaffNet and FusionNet <a name="running-scaffnet-fusionnet"></a>
To run our pretrained ScaffNet on the VOID dataset, you may use
```
bash bash/run_scaffnet-void1500.sh
```

To run our pretrained FusionNet on the VOID dataset, you may use
```
bash bash/run_fusionnet-void1500.sh
```

You may replace the restore_path and output_path arguments to evaluate your own checkpoints


## Training ScaffNet and FusionNet <a name="training-scaffnet-fusionnet"></a>
To train ScaffNet on the Virtual KITTI dataset, you may run
```
sh bash/train_scaffnet-vkitti.sh
```

To train ScaffNet on the SceneNet dataset, you may run
```
sh bash/train_scaffnet-scenenet.sh
```

To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_scaffnet/vkitti/<model_name>
tensorboard --logdir trained_scaffnet/scenenet/<model_name>
```

Unlike Tensorflow version of this repository, we will not need a separate set up script to generate ScaffNet predictions for training FusionNet. We will instead load in weights of ScaffNet, freeze them and learn its residual using FusionNet. Note that our PyTorch training code directly uses ScaffNet as part of the inference pipeline end-to-end improves performance. See table above.

To train FusionNet on the KITTI dataset, you may run
```
sh bash/train_fusionnet-kitti.sh
```

To train FusionNet on the VOID dataset, you may run
```
sh bash/train_fusionnet-void1500.sh
```

To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_fusionnet/kitti/<model_name>
tensorboard --logdir trained_fusionnet/void/<model_name>
```

## Related projects <a name="related-projects"></a>
You may also find the following projects useful:

- [MonDi][mondi_github]: *Monitored Distillation for Positive Congruent Depth Completion (MonDi)*. A method for blind ensemble distillation that leverages a monitoring validation function to allow student models trained through the distillation process to retain strengths of teachers while minimizing distillation of their weaknesses. This work is published in the European Conference on Computer Vision (ECCV) 2022.
- [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*. A fast (15 ms/frame) and accurate unsupervised sparse-to-dense depth completion method that introduces a calibrated backprojection layer that improves generalization across sensor platforms. This work is published as an *oral* paper in the International Conference on Computer Vision (ICCV) 2021.
- [ScaffNet][scaffnet_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An unsupervised sparse-to-dense depth completion method that first learns a map from sparse geometry to an initial dense topology from synthetic data (where ground truth comes for free) and amends the initial estimation by validating against the image. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [AdaFrame][adaframe_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An adaptive framework for learning unsupervised sparse-to-dense depth completion that balances data fidelity and regularization objectives based on model performance on the data. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [VOICED][voiced_github]: *Unsupervised Depth Completion from Visual Inertial Odometry*. An unsupervised sparse-to-dense depth completion method, developed by the authors. The paper introduces Scaffolding for depth completion and a light-weight network to refine it. This work is published in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [VOID][void_github]: from *Unsupervised Depth Completion from Visual Inertial Odometry*. A dataset, developed by the authors, containing indoor and outdoor scenes with non-trivial 6 degrees of freedom. The dataset is published along with this work in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab. This work is built on top of XIVO. The VOID dataset used by this work also leverages XIVO to obtain sparse points and camera poses.
- [GeoSup][geosup_github]: *Geo-Supervised Visual Depth Prediction*. A single image depth prediction method developed by the authors, published in the Robotics and Automation Letters (RA-L) 2019 and the International Conference on Robotics and Automation (ICRA) 2019. This work was awarded **Best Paper in Robot Vision** at ICRA 2019.
- [AdaReg][adareg_github]: *Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.* A single image depth prediction method that introduces adaptive regularization. This work was published in the proceedings of Conference on Computer Vision and Pattern Recognition (CVPR) 2019.

We also have works in adversarial attacks on depth estimation methods and medical image segmentation:
- [SUPs][sups_github]: *Stereoscopic Universal Perturbations across Different Architectures and Datasets..* Universal advesarial perturbations and robust architectures for stereo depth estimation, published in the Proceedings of Computer Vision and Pattern Recognition (CVPR) 2022.
- [Stereopagnosia][stereopagnosia_github]: *Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.* Adversarial perturbations for stereo depth estimation, published in the Proceedings of AAAI Conference on Artificial Intelligence (AAAI) 2021.
- [Targeted Attacks for Monodepth][targeted_attacks_monodepth_github]: *Targeted Adversarial Perturbations for Monocular Depth Prediction.* Targeted adversarial perturbations attacks for monocular depth estimation, published in the proceedings of Neural Information Processing Systems (NeurIPS) 2020.
- [SPiN][spin_github] : *Small Lesion Segmentation in Brain MRIs with Subpixel Embedding.* Subpixel architecture for segmenting ischemic stroke brain lesions in MRI images, published in the Proceedings of Medical Image Computing and Computer Assisted Intervention (MICCAI) Brain Lesion Workshop 2021 as an **oral paper**.

[kitti_dataset]: http://www.cvlibs.net/datasets/kitti/
[vkitti_dataset]: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
[scenenet_dataset]: https://robotvault.bitbucket.io/scenenet-rgbd.html
[nyu_v2_dataset]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
[void_github]: https://github.com/alexklwong/void-dataset
[voiced_github]: https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry
[scaffnet_github]: https://github.com/alexklwong/learning-topology-synthetic-data
[adaframe_github]: https://github.com/alexklwong/adaframe-depth-completion
[kbnet_github]: https://github.com/alexklwong/calibrated-backprojection-network
[mondi_github]: https://github.com/alexklwong/mondi-python
[xivo_github]: https://github.com/ucla-vision/xivo
[geosup_github]: https://github.com/feixh/GeoSup
[adareg_github]: https://github.com/alexklwong/adareg-monodispnet
[sups_github]: https://github.com/alexklwong/stereoscopic-universal-perturbations
[stereopagnosia_github]: https://github.com/alexklwong/stereopagnosia
[targeted_attacks_monodepth_github]: https://github.com/alexklwong/targeted-adversarial-perturbations-monocular-depth
[spin_github]: https://github.com/alexklwong/subpixel-embedding-segmentation

## License and disclaimer <a name="license-disclaimer"></a>
This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](license). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).
