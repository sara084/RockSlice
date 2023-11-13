# RockS2Net

**RockS2Net is a python project that is a Siamese network for rock classification**

## Installation

`pip install tensorflow==2.2.0`

`pip install numpy==1.19.5`

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
| RockS2Net | 87.14%    | 91.92% |    97.75%   | 92.14% | 91.85% |
