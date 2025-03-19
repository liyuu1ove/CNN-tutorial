From compose dataset to deploy on UAVs

# Train DNN model (take Yolov8 for an example)
## Setup CUDA environment (Nvidia GPU required,better if with 10GB+ video memory )
* install CUDA
  ```bash
  #check adoptable cuda verison
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
  cd path/to/cuda/demo_suite # for example C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\extras\demo_suite
  .\bandwidthTest.exe

  ```
  output
  ```
  [CUDA Bandwidth Test] - Starting...
  Running on...

  Device 0: NVIDIA GeForce RTX 4080 Laptop GPU
  Quick Mode

  Host to Device Bandwidth, 1 Device(s)
  PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(MB/s)
  33554432                     12707.6

  Device to Host Bandwidth, 1 Device(s)
  PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(MB/s)
  33554432                     12803.5

  Device to Device Bandwidth, 1 Device(s)
  PINNED Memory Transfers
  Transfer Size (Bytes)        Bandwidth(MB/s)
  33554432                     149433.4

  Result = PASS
  ```
## Dependent installation
* Create isolated conda envs
  ```shell
  $conda shell:
  (base)conda create -n Yolov8 python=3.8
  ```
* Activate environment
  ```shell
  $conda shell:
  (base)conda activate Yolov8
  (Yolov8)
* install [pytorch](https://pytorch.org/get-started/locally/)
  ```shell
  $conda shell:
  # select your vision on the website!
  (Yolov8) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
  

* install ultralytics

  yolo core code is packed in ultralytics lib  
  ```shell
  $conda shell
  (Yolov8) cd /path/to/Yolo 
  (Yolov8) pip install ultralytics
  ```

* clone ultralytics git repo
  ```bash
  $git bash
  git clone https://github.com/ultralytics/ultralytics
  ```
  all models are included in the repo,so just clone the newest one


## Train
### Building data sets(standard yolo format)
* Labelimg
  download labelimg on [labelimg](https://github.com/HumanSignal/labelImg)

 * **build labelimg** on windows
    ```shell
    $conda shell
    (base)conda create -n Labelimg python=3.8
    (base)conda activate Labelimg
    (Labelimg)conda install pyqt=5
    (Labelimg)conda install -c anaconda lxml
    (Labelimg)cd path/to/labelimg #change to you dir
    (Labelimg)pyrcc5 -o libs/resources.py resources.qrc
    (Labelimg)python labelImg.py  #run labelImg
    (Labelimg)python labelImg.py [path/to/images] [path/to/prebuild/label.txt] #todo find save dir
    ```

* The procedure to create train/val/test files is automated by using **generate_dataset.py** 
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
  
### Build the training dataset.yaml configuration file
* Reference ball.yaml
  ```
  path: ./dataset # dataset root dir
  train: train.txt # train images (relative to 'path')
  val: val.txt # val images (relative to 'path')
  test: test.txt # test images (relative to 'path')

  # Classes
  names:
    0: person
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
  ```
### Train
* modify yolov8.yaml in ultralytics git repo at *ultralytics\ultralytics\cfg\models\v8*
  ```
  ...
  nc:6 #change the number to match your dataset.yaml
  ...
  #no other change needed
  ```
* Perform training tasks in CLI
  ```shell
  $conda shell
  (Yolov8)path/to/ultralytics> yolo task=detect mode=train model=yolov8s.yaml pretrained= yolov8s.pt data=dataset.yaml epochs=300 batch=16
  ```
* Perform training tasks using Python API
  ```bash
  $bash
  python train.py#change parameters in train.py
  ```
  the parameter *model* will call yolov8n if you use the name yolov8n.yaml 
  
  using pretrained model to enhance the performance of your model, the pretrained model will be downloaded automatically when you call the model
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
* 
  ```
  ```

*This instruction is written by Fangyao Zhao at HUST/Berkeley nicknamed as liyuu1ove on github,following the MIT license,please be careful when you spread it*