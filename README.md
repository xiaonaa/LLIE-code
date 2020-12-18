# LLIENet

Estimated Exposure Guided Reconstruction Model for Low-Light Image Enhancement. In PRCV2020

Xiaona Liu, Yang Zhao, Yuan Chen, Wei Jia , Ronggang Wang, and Xiaoping Liu 

![result1](https://github.com/xiaonaa/LLIE-code/blob/master/figures/result1.png)

## RequireMents

1.Python

2.PyTorch 1.0.0

3.numpy

Note: If your pytorch version is higher than 1.0 but lower than or equal to 1.5, please replace the dataloader.py and utility.py files under `./code`. The replacement files are placed under `./py15`.

## Usage

**Testing**

First download the pre-training models from [BaiduNetdisk](https://pan.baidu.com/s/1a9GWHHpLWI1v3PWbneDX8Q ) (qx2r) and put them under `./experiment/test/`.  If you want to use your own testing set, you should put your testing set under `./train/benchmark/`. Then to quickly test your own low-light images with our model, you can just run through

```
python option.py
	--test_only=true
	--resume=300
	--save_results=true
python trainer.py
```

Note that our test image must be a multiple of 16. The results will be saved under `./experiment/test/results/`.

**Training**

If you want to use your own training set for training, you should put your training set under `./train/`. Then, just run

```
python option.py
	--test_only 
	--resume=0
	--save_results=true
python trainer.py
```

## Acknowledgements

This code is built on  [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch).

