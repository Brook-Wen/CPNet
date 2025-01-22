# CPNet

## [Consistency perception network for 360° omnidirectional salient object detection](https://www.sciencedirect.com/science/article/pii/S0925231224020149)

This repo is an official implementation of the *CPNet*, which has been accepted by the journal *Neurocomputing, 2025*.

![CPNet](Figures/CPNet.bmp)

![V-SOTA](Figures/V-SOTA.bmp)


## Prerequisites

- python=3.x
- pytorch=1.0.0+
- torchvision
- numpy
- opencv-python


## Usage

### 1. Clone the repository

```
git clone https://github.com/Brook-Wen/CPNet.git
cd CPNet/
```

### 2. Training

```
python main.py --mode='train'
```

### 3. Testing

```
python main.py --mode='test' --batch_size 1 --model='[YOUR PATH]' --test_fold='[SAVE PATH]' --sal_mode='[DATASET]'
```

- We provide the trained [model weights](https://pan.baidu.com/s/1oZYqf1gpVnCzSv00rZ_GPw) (fetch code: fqft). 

### 4. Evaluation

- We provide the predicted [saliency maps](https://pan.baidu.com/s/1inWoy9TemXWkUhVPn1PvqQ) (fetch code: xepc) of our CPNet on three datasets.
- You can use this [toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluation.


## Citation

### If you think this work is helpful, please cite
```
@inproceedings{wen2025cpnet,
  title={Consistency perception network for 360° omnidirectional salient object detection},
  author={Wen, Hongfa and Zhu, Zunjie and Zhou, Xiaofei and Zhang, Jiyong and Yan, Chenggang},
  booktitle={Neurocomputing},
  year={2025}
}
```

- If you have any questions, feel free to contact me via: `hf_wen(at)outlook.com`.
