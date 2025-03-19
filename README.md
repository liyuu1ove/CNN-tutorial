From compose dataset to deploy on UAVs
# Train DNN model (take Yolov8 for an example)
## Setup CUDA environment (Nvidia GPU required,better if with 10GB+ video memory )
* install CUDA
  ```bash
  #check cuda verison
  $bash
  nvidia-smi
  ```
  find required version on [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
* install cuDNN

  select cuDNN version base on CUDA version on [NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

  extract cuDNN and cut **bin,include,lib** to where you install CUDA, for example C:/program files/NVIDIA GPU Computing Toolkit/CUDA/12.5

* check environment
  ```bash
  $bash
  cd /test
  python test_cuda_env.py
  ```

## Download Yolo code
* see [ultralytics](https://docs.ultralytics.com/zh)
* Yolov8 on [Yolov8](https://docs.ultralytics.com/models/yolov8/)
## Dependent installation
* Create isolated conda envs
  ```shell
  $conda shell:
  (base)conda create -n Yolov8 python=X.X
  ```
* Activate environment
  ```shell
  $conda shell:
  (base)conda activate Yolov8
  (Yolov8)
* install [pytorch](https://pytorch.org/get-started/locally/)
  ```shell
  $conda shell:
  (Yolov8) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
  # select your vision on the website!

* PiP(change pytorch CUDA version in **requirements.txt** to your vision)
  ```shell
  $conda shell
  (Yolov8) cd /path/to/Yolo 
  (Yolov8) pip install -r requirements.txt
  ```


## Train
### Building data sets(standard yolo format)
* Labelimg
* The procedure is automated by using **generate_dataset.py** 
  ```bash
  $bash:
  python generate_dateset
  ```
* The format of the data set is known as Darknet Yolo, Each image corresponds to a .txt label file. The label format is based on Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  dataset
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path(use absolute path) .txt file, the example content is as follows：
  
  train.txt
  ```
  C:/Desktop/Yolov8/dataset/train/000001.jpg
  C:/Desktop/Yolov8/dataset/train/000002.jpg
  C:/Desktop/Yolov8/dataset/train/000003.jpg
  ```
  val.txt
  ```
  C:/Desktop/Yolov8/dataset/val/000070.jpg
  C:/Desktop/Yolov8/dataset/val/000043.jpg
  C:/Desktop/Yolov8/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:
 
  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Build the training .yaml configuration file
* Reference./configs/coco.yaml
  ```
  DATASET:
    TRAIN: "/home/qiuqiu/Desktop/coco2017/train2017.txt"  # Train dataset path .txt file
    VAL: "/home/qiuqiu/Desktop/coco2017/val2017.txt"      # Val dataset path .txt file 
    NAMES: "dataset/coco128/coco.names"                   # .names category label file
  MODEL:
    NC: 80                                                # Number of detection categories
    INPUT_WIDTH: 352                                      # The width of the model input image
    INPUT_HEIGHT: 352                                     # The height of the model input image
  TRAIN:
    LR: 0.001                                             # Train learn rate
    THRESH: 0.25                                          
    WARMUP: true                                          # Trun on warm up
    BATCH_SIZE: 64                                        # Batch size
    END_EPOCH: 350                                        # Train epichs
    MILESTIONES:                                          # Declining learning rate steps
      - 150
      - 250
      - 300
  ```
### Train
* Perform training tasks
  ```
  $conda shell
  (Yolov8)yolo task=detect mode=train model=yolov8s.pt data=ball.yaml epochs=1000 batch=16
  ```
### Evaluation 
* Calculate map evaluation
  ```shell
  
  ```
* test on test.txt
  ```shell

  ```
* Predict
  ```shell
  $conda shell
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg
  ```
# Deploy
## Export onnx
* You can export .onnx by adding the --onnx option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --onnx
  ```
## Export torchscript
* You can export .pt by adding the --torchscript option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --torchscript
  ```
## NCNN
* Need to compile ncnn and opencv in advance and modify the path in build.sh
  ```
  cd example/ncnn/
  sh build.sh
  ./FastestDet
  ```
## onnx-runtime
* You can learn about the pre and post-processing methods of FastestDet in this Sample
  ```
  cd example/onnx-runtime
  pip install onnx-runtime
  python3 runtime.py
  ```
This instruction is written by Fangyao Zhao at HUST,following the MIT li