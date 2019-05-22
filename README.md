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

1. Extract features from network you want to use, and locate them at `<PROJECT ROOT>/<DATASET>/features/<DATASET>_<NETWORK>.hdf5`. I extracted features from [here](https://github.com/hobincar/video-feature-extractor) for VGG19, ResNet-152, Inception-v4, DenseNet, and ShuffleNet, [here](https://github.com/facebook/C3D) for C3D and Res3D, and [here](https://github.com/facebookresearch/VMZ) for R(2+1)D network.

2. Split the dataset along with the official splits after changing `model` of `<DATASET>SplitConfig` in `config.py`, and run following:

   ```
   (.env) $ python -m splits.MSVD
   (.env) $ python -m splits.MSR-VTT
   ```


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

1. Set the checkpoint path by changing `ckpt_fpath` of `EvalConfig` in `config.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


# Performances

I select a checkpoint which achieves the best CIDEr score on the validation set, and report the test scores of it. All experiments are run 5 times and averaged. For SqueezeNet [7], I met a memory issue because the size of feature vector is 86528.

* MSVD

  | Model | Features | Trained on | BLEU4 | CIDEr | METEOR | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM [1] | GoogLeNet [2] & 3D conv. | | 41.92 | 51.67 | 29.6 | - |
  | SA-LSTM [3] | Inception-v4 [4] | ImageNet | 45.3 | 76.2 | 31.9 | 64.2 |
  |  |  |  |  |  |
  | Ours | VGG19 [5] | ImageNet | 46.37	| 68.27 |	31.16 |	67.37 |
  | Ours | ResNet-152 [6] | ImageNet | 50.84	| **79.47** |	33.25 |	69.81 |
  | Ours | Inception-v4 [4] | ImageNet | 50.2	| 79.04 |	33.3 |	69.65 |
  | Ours | DenseNet [8] | ImageNet | 49.1	| 73.28 | 32.6 |	69.23 |
  | Ours | ShuffleNet [9] | ImageNet | 35.04 |	42.9 |	25.9 |	62.62 |
  | Ours | C3D fc6 [10] | Sports1M | 44.32	| 53.42 |	29.66 |	67.05 |
  | Ours | C3D fc7 [10] | Sports1M | 45.53 |	59.4 |	30.71 |	67.79 |
  | Ours | Res3D res5b [11] | Sports1M | - | - | - | - |
  | Ours | Res3D pool5 [11] | Sports1M | - | - | - | - |
  | Ours | R(2+1)D [12] | Sports1M, finetuned on Kinetics | **51.23**	| 77.81 |	**33.44** |	**70.06** |


* MSR-VTT

  | Model | Features | Trained on | BLEU4 | CIDEr | METEOR | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM [3] | Inception-v4 | ImageNet | 36.3 | 39.9 | **25.5** | **58.3** |
  |  |  |  |  |  |
  | Ours | VGG19 [5] | ImageNet | 34.94	| 37.45 |	24.62 |	56.33 |
  | Ours | ResNet-152 [6] | ImageNet | 36.4 |	41.32 |	**25.51** |	57.57 |
  | Ours | Inception-v4 [4] | ImageNet | 36.24	| 40.89 |	25.25 |	57.31 |
  | Ours | DenseNet [8] | ImageNet | 36.2 |	40.24 |	25.32 |	57.34 |
  | Ours | ShuffleNet [9] | ImageNet | 34.04	| 34.11 |	23.89 |	55.78 |
  | Ours | C3D fc6 [10] | Sports1M | - | - | - | - |
  | Ours | C3D fc7 [10] | Sports1M | - | - | - | - |
  | Ours | Res3D res5b [11] | Sports1M | - | - | - | - |
  | Ours | Res3D pool5 [11] | Sports1M | - | - | - | - |
  | Ours | R(2+1)D [12] | Sports1M, finetuned on Kinetics | **36.72** |	**41.42** |	25.35 |	57.72 |


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

[11] Tran, Du, et al. "Convnet architecture search for spatiotemporal feature learning." arXiv preprint arXiv:1708.05038 (2017).

[12] Tran, Du, et al. "A closer look at spatiotemporal convolutions for action recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.
