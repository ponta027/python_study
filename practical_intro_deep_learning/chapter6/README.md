# Chapter 6 画像処理


## Setup

```bash
>pip install opencv-python
```

ImportError: libGL.so.1: cannot open shared object file: No such file or directory

```bash
> sudo apt-get install libgl1-mesa-dev
```

### Download Data

```bash
> wget https://drive.google.com/open?id=1n0fI1A-KMq5PCQTje1M088b_S3hh4dDb
```


## torchvisionのインストール

MNISTの動作確認をする際にtorchvisionをインストールする。

```bash
> pip install torchvision
```

2023-01-08現在だと0.14.1がインストールされる。
torchivisonをimportすると例外が発生する。

```bash
python 3.7.16 (default, Dec 21 2022, 11:39:51) 
[GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torchvision
/usr/local/lib/python3.7/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.7/site-packages/torchvision/__init__.py", line 5, in <module>
    from torchvision import datasets, io, models, ops, transforms, utils
  File "/usr/local/lib/python3.7/site-packages/torchvision/models/__init__.py", line 17, in <module>
    from . import detection, optical_flow, quantization, segmentation, video
  File "/usr/local/lib/python3.7/site-packages/torchvision/models/quantization/__init__.py", line 3, in <module>
    from .mobilenet import *
  File "/usr/local/lib/python3.7/site-packages/torchvision/models/quantization/mobilenet.py", line 1, in <module>
    from .mobilenetv2 import *  # noqa: F401, F403
  File "/usr/local/lib/python3.7/site-packages/torchvision/models/quantization/mobilenetv2.py", line 5, in <module>
    from torch.ao.quantization import DeQuantStub, QuantStub
ModuleNotFoundError: No module named 'torch.ao'
```

バージョンの不整合と思われるので、バージョンを下げる。
※時間があったら根本原因をか調査する。



```bash 
>pip install torchvision==0.9.1
```

でimportしたら問題が発生しなかった。



## 実行時のスレッド数

DataLoaderを呼び出すと下記警告が表示される。

```bash
/usr/local/lib/python3.7/site-packages/pytorch_lightning/utilities/distributed.py:45: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 4 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  warnings.warn(*args, **kwargs)
/usr/local/lib/python3.7/site-packages/pytorch_lightning/utilities/distributed.py:45: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 4 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
```

https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
のnum_workersを調整が必要とのこと。
動作しているプログラムは10パラレル。
動作環境のコアは4コア

そのため、ボトルネックになっている



