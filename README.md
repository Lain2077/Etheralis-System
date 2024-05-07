# Etheralis-System
This repo is the core system of Etheralis, an immersive project taking advantages of a realtime dynamic face recognition system performant on consumer grade hardware.

## Description
This custom implementation of a facenet_pytorch allows a dynamically updatable database to constantly be compared in batch with people present in a given webcam's feed.

### Modules used
#### Essentials to facerec & communicate data
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [pytorch](https://github.com/pytorch/pytorch)
- [torchvision](https://github.com/pytorch/pytorch)
- [ndi-python (Python 3.11.4 build)](https://github.com/buresu/ndi-python)
- [opencv-python (CUDA support is optional)](https://github.com/opencv/opencv-python)
- [python-osc](https://github.com/attwad/python-osc)
- [numpy]()
- [glob]()
- [threading]()
- [PIL]()
- Included custom implementation of the Fast_MTCNN algorithm
- Included database updater

#### Specific usecase
- [requests]()


## Getting it running
### Environment setup

This code has been developped on Python 3.11.4 with CUDA 12.1. To make sure everything works accordingly, setup an environment using these specified versions.

#### Install specific pytorch modules version
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Build and install ndi-python for Python 3.11.4

In order to run NDIlib on Python 3.11.4, you need to build it yourself.

1. Clone the repository to a local directory and cd to it

2. ```python setup.py bdist_wheel```

3. ```python setup.py install```


#### OpenCV

More often than not, although depending on your setup, you might take advantages of OpenCV being built with cuda support. If you want to give it a try, I used a prebuilt wheel from [this repo](https://github.com/cudawarped/opencv-python-cuda-wheels). I used the 4.7.0.20230527 release.

1. ```pip install wheel``` and download corresponding wheel from repo
2. Navigate to download folder and pip install wheel_file_name.whl

#### Others
For every other modules, the provided requirements.txt should do the job.

A RAW_requirements.txt is also included, brutally installing every needed modules. Expect errors if you don't know what you're doing