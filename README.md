# SA-LSTM

This project tries to implement *SA-LSTM* proposed by **[Describing Videos by Exploiting Temporal Structure](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf)**[1], *ICCV 2015*.



# Environment

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Nvidia Geforce GTX Titan Xp 12GB


# Requirements

* Java 8
* Python 2.7.12
  * PyTorch 1.0
  * Other python libraries specified in requirements.txt



# How to use

## Step 1. Setup python virtual environment

```
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```


## Step 2. Prepare Data

1. Download feature vectors for each dataset ([MSVD](https://github.com/hobincar/MSVD), [MSR-VTT](https://github.com/hobincar/MSR-VTT)), and locate them at `~/<dataset>/features/<network>_<phase>.hdf5`. For example, InceptionV4 feature vectors for MSVD train dataset will be located at `~/data/MSVD/features/InceptionV4_train.hdf5`.


## Step 3. Prepare Evaluation Codes

Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```


## Step 4. Train

Run
   ```
   (.env) $ python train.py
   ```

You can change some hyperparameters by modifying `config.py`.


## Step 5. Inference

1. Set the checkpoint path by changing the value of a property named `ckpt_fpath` of `EvalConfig` in `config.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


# Performances

I select a checkpoint which achieves the best CIDEr score on the validation set, and report the test scores of it. All experiments are run 5 times and averaged.

* MSVD

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM[1] | GoogLeNet[2], 3D conv. (HOG+HOF+MBH) | 41.92 | 29.6 | 51.67 | - |
  | SA-LSTM[3] | Inception-v4[4] | 45.3 | 31.9 | **76.2** | 64.2 |
  |  |  |  |  |
  | Ours | VGG19[5] | - | - | - | - |
  | Ours | ResNet-50[6] | - | - | - | - |
  | Ours | ResNet-101[6] | - | - | - | - |
  | Ours | ResNet-152[6] | - | - | - | - |
  | Ours | Inception-v4[4] | - | - | - | - |
  | Ours | SqueezeNet[7] | - | - | - | - |
  | Ours | DenseNet[8] | - | - | - | - |
  | Ours | ShuffleNet[9] | - | - | - | - |
  | Ours | C3D (Sports1M)[10] | - | - | - | - |
  | Ours | R2.5D (Kinetics)[11] | - | - | - | - |
  | Ours | R2.5D (Sports1M + Kinetics)[11] | - | - | - | - |


* MSR-VTT

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM[3] | Inception-v4 | 36.3 | 25.5 | 39.9 | **58.3** |
  |  |  |  |  |
  | Ours | VGG19[5] | - | - | - | - |
  | Ours | ResNet-50[6] | - | - | - | - |
  | Ours | ResNet-101[6] | - | - | - | - |
  | Ours | ResNet-152[6] | - | - | - | - |
  | Ours | Inception-v4[4] | - | - | - | - |
  | Ours | SqueezeNet[7] | - | - | - | - |
  | Ours | DenseNet[8] | - | - | - | - |
  | Ours | ShuffleNet[9] | - | - | - | - |
  | Ours | C3D (Sports1M)[10] | - | - | - | - |
  | Ours | R2.5D (Kinetics)[11] | - | - | - | - |
  | Ours | R2.5D (Sports1M + Kinetics)[11] | - | - | - | - |


# References

[1] Yao, Li, et al. "Describing videos by exploiting temporal structure." Proceedings of the IEEE international conference on computer vision. 2015.

[2] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[3] Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[4] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." AAAI. Vol. 4. 2017.

[5] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[6] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[7] Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv preprint arXiv:1602.07360 (2016).

[8] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[9] Zhang, Xiangyu, et al. "Shufflenet: An extremely efficient convolutional neural network for mobile devices." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[10] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.

[11] Tran, Du, et al. "A closer look at spatiotemporal convolutions for action recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.
