# RockS2Net

**RockS2Net is a python project that is a Siamese network for rock classification**

## Installation

`pip install tensorflow==2.2.0`

`pip install numpy==1.19.5`

`pip install -r requirements.txt`
## Requirements
- confusion-matrix==0.1

- Keras==2.3.1

- matplotlib==3.5.2

- numpy==1.19.5

- Pillow==9.3.0

- pip==22.3.1

- scikit-learn==1.0.2

- scipy==1.7.3

- sklearn==0.0.post1

## Data acquisition
Visit the following link and enter the password to download：

`https://pan.baidu.com/s/1xZJ9KzfCGmJ6_nsB8dYiVg`

`password: 1wj6`


## Usage
1、You can change the datasets and related data here to set different tasks：

`train_generator=datagen.flow_from_directory('/home/train_data',target_size=(512,512),batch_size=4, save_format='jpg')`

`test_generator=datagen.flow_from_directory('/home/test_data',target_size=(512,512),batch_size=4, save_format='jpg')`

2、You can add weight files here：

`7z densenet121_weights_tf_dim_ordering_tf_kernels_notop.7z -r -o /home`

`weight_path="/home/.h5"`

`model.load_weights('\home\.h5')`

3、Run `python STNdensenet121.py`.

## Result

| Method |   Grain   | Clastic | Mechanical genesis | Mixture | Basic category |
|:--------------:|:---------:|:-----:|:------------------:|:-----:|:----:|
| RockS2Net（Ours）| 87.14%    | 91.92% |    97.75%   | 92.14% | 91.85% |
| ResNet34 | 85.34%    | 88.79% |    93.88%   | 90.23% | 87.99% |
| DarkNet53 | 86.12%    | 89.91% |    92.54%   | 91.67% | 88.83% |
| EfficientNetB0 | 86.91%    | 90.42% |    96.69%   | 91.94% | 89.96% |

## Eval time

| Method |   Grain（ms）   | Clastic（ms） | Mechanical genesis（ms） | Mixture（ms） | Basic category（ms） |
|:--------------:|:---------:|:-----:|:------------------:|:-----:|:----:|
|RockS2Net（Ours）| 320.11    | 283.11 |    314.89   | 298.67 | 289.44 |

