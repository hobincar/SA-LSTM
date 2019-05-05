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
  | SA-LSTM[3] | InceptionV4[4] | 45.3 | 31.9 | **76.2** | 64.2 |
  |  |  |  |  |
  | Ours | VGG19 | - | - | - | - |
  | Ours | ResNet-50 | - | - | - | - |
  | Ours | ResNet-101 | - | - | - | - |
  | Ours | ResNet-152 | - | - | - | - |
  | Ours | Inception-v4 | - | - | - | - |


* MSR-VTT

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM[3] | InceptionV4 | 36.3 | 25.5 | 39.9 | **58.3** |
  | Ours | InceptionV4 | **37.29** | **26.00** | **42.6** | 58.13 |
  |  |  |  |  |
  | Ours | VGG19 | - | - | - | - |
  | Ours | ResNet-50 | - | - | - | - |
  | Ours | ResNet-101 | - | - | - | - |
  | Ours | ResNet-152 | - | - | - | - |
  | Ours | Inception-v4 | - | - | - | - |


# References

[1] Yao, Li, et al. "Describing videos by exploiting temporal structure." Proceedings of the IEEE international conference on computer vision. 2015.

[2] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[3] Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[4] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." AAAI. Vol. 4. 2017.
