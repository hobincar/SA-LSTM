# SA-LSTM

This project tries to implement *SA-LSTM* proposed in **[Describing Videos by Exploiting Temporal Structure](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf) [1], *ICCV 2015***.



## Environment

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Nvidia Geforce GTX Titan Xp 12GB


## Requirements

* Java 8
* Python 2.7.12
  * PyTorch 1.0
  * Other python libraries specified in requirements.txt



## How to use

### Step 1. Setup python virtual environment

```
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```


### Step 2. Prepare Data

1. Extract features from network you want to use, and locate them at `<PROJECT ROOT>/<DATASET>/features/<DATASET>_<NETWORK>.hdf5`. I extracted features from [here](https://github.com/hobincar/pytorch-video-feature-extractor) for VGG19, ResNet-152, Inception-v4, DenseNet, and ShuffleNet, [here](https://github.com/facebook/C3D) for C3D, and [here](https://github.com/facebookresearch/VMZ) for R(2+1)D network.

2. Split the dataset along with the official splits after changing `model` of `<DATASET>SplitConfig` in `config.py`, and run following:

   ```
   (.env) $ python -m splits.MSVD
   (.env) $ python -m splits.MSR-VTT
   ```


### Step 3. Prepare Evaluation Codes

Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```


### Step 4. Train

Run
   ```
   (.env) $ python train.py
   ```

You can change some hyperparameters by modifying `config.py`.


### Step 5. Inference

1. Set the checkpoint path by changing `ckpt_fpath` of `EvalConfig` in `config.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


## Results

I select a checkpoint which achieves the best CIDEr score on the validation set, and report the test scores of it. All experiments are run 5 times and averaged. For SqueezeNet [7], I met a memory issue because the size of feature vector is 86528.

* MSVD

  | Model | Features | Trained on | BLEU4 | CIDEr | METEOR | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM [1] | GoogLeNet [2] & 3D conv. | | 41.92 | 51.67 | 29.6 | - |
  | SA-LSTM [3] | Inception-v4 [4] | ImageNet | 45.3 | 76.2 | 31.9 | 64.2 |
  |  |  |  |  |  |
  | Ours | AlexNet [12] | ImageNet | 36.3 |	34.9 |	26.7 |	63.4 |
  | Ours | GoogleNet [13] | ImageNet | 36.0 |	38.8 |	25.0 |	57.1 |
  | Ours | VGG19 [5] | ImageNet | 46.4	| 68.3 |	31.2 |	67.4 |
  | Ours | ResNet-152 [6] | ImageNet | 50.8	| 79.5 |	33.3 |	69.8 |
  | Ours | ResNext-101 [14] | ImageNet | 50.0 |	77.2 |	33.0 |	63.4 |
  | Ours | Inception-v4 [4] | ImageNet | 50.2	| 79.0 |	33.3 |	69.7 |
  | Ours | DenseNet [8] | ImageNet | 49.4 |	75.6 |	32.7 |	69.2 |
  | Ours | ShuffleNet [9] | ImageNet | 35.0 |	42.9 |	25.9 |	62.6 |
  | Ours | C3D fc6 [10] | Sports1M | 44.3	| 53.4 |	29.7 |	67.1 |
  | Ours | C3D fc7 [10] | Sports1M | 45.5 |	59.4 |	30.7 |	67.8 |
  | Ours | R(2+1)D [11] | Sports1M, finetuned on Kinetics | **51.2**	| 77.8 |	**33.4** |	**70.1** |
  | Ours | 3D-ResNext-101 [15] | Kinetics | 49.2	| **82.3** |	33.1 |	70.0 |


* MSR-VTT

  | Model | Features | Trained on | BLEU4 | CIDEr | METEOR | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM [3] | Inception-v4 | ImageNet | 36.3 | 39.9 | **25.5** | **58.3** |
  |  |  |  |  |  |
  | Ours | AlexNet [12] | ImageNet | 31.3 |	29.8 |	23.3 |	54.5 |
  | Ours | GoogleNet [13] | ImageNet | 26.5 |	26.0 |	22.4 |	58.4 |
  | Ours | VGG19 [5] | ImageNet | 34.9	| 37.4 |	24.6 |	56.3 |
  | Ours | ResNet-152 [6] | ImageNet | 36.4 |	41.3 |	25.5 |	57.6 |
  | Ours | ResNext-101 [14] | ImageNet | 36.5 |	**41.9** |	**25.7** |	**57.8** |
  | Ours | Inception-v4 [4] | ImageNet | 36.2	| 40.9 |	25.3 |	57.3 |
  | Ours | DenseNet [8] | ImageNet | 36.8 |	40.9 |	25.6 |	57.9 |
  | Ours | ShuffleNet [9] | ImageNet | 34.0	| 34.1 |	23.9 |	55.8 |
  | Ours | C3D fc6 [10] | Sports1M | - | - | - | - |
  | Ours | C3D fc7 [10] | Sports1M | - | - | - | - |
  | Ours | R(2+1)D [11] | Sports1M, finetuned on Kinetics | **36.7** |	41.4 |	25.4 |	57.7 |
  | Ours | 3D-ResNext-101 [15] | Kinetics | -	| - |	- |	- |


## References

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

[12] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

[13] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[14] Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[15] Hara, Kensho, Hirokatsu Kataoka, and Yutaka Satoh. "Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet?." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.
