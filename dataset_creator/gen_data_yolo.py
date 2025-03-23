
import os
import random
import shutil
import time
import cv2
import numpy as np

val_ratio = 0.1
test_ratio = 0.1
#best pratice 8:1:1 but you can shrink 
#the test ratio when data is reletively small
# target_size = (360, 480)
target_size = None

data_dir = "./dataset/data"
train_dir = "./dataset/train"
val_dir = "./dataset/val"
test_dir="./dataset/test"

train_file = "./dataset/train.txt"
val_file = "./dataset/val.txt"
test_file="./dataset/test.txt"
classes_file = "./dataset/category.names"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
else:
    shutil.rmtree(train_dir)
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
else:
    shutil.rmtree(val_dir)
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
else:
    shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
filelist = os.listdir(data_dir)

imglist = [file for file in filelist if file.endswith(".jpg")]

random.seed(time.time())
random.shuffle(imglist)

train_list = []
val_list = []
test_list = []

count = 0

for img in imglist:
    count += 1
    if (count <= int((val_ratio+test_ratio) * len(imglist))): # 验证集和测试集
        if (count <= int(test_ratio * len (imglist))):#测试集
            test_list.append(img)
            if target_size is None:
                shutil.copy(os.path.join(data_dir, img), os.path.join(test_dir, img))
            else:
                img = cv2.imread(os.path.join(data_dir, img))
                img = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(test_dir, img), img)
            txt_file = img.replace(".jpg", ".txt")
            shutil.copy(os.path.join(data_dir, txt_file), os.path.join(test_dir, txt_file))
            print("copy {} to {}".format(img, test_dir))
        else:
            val_list.append(img)
            if target_size is None:
                shutil.copy(os.path.join(data_dir, img), os.path.join(val_dir, img))
            else:
                img = cv2.imread(os.path.join(data_dir, img))
                img = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(val_dir, img), img)
            txt_file = img.replace(".jpg", ".txt")
            shutil.copy(os.path.join(data_dir, txt_file), os.path.join(val_dir, txt_file))
            print("copy {} to {}".format(img, val_dir))
    else: # 训练集
        train_list.append(img)
        if target_size is None:
            shutil.copy(os.path.join(data_dir, img), os.path.join(train_dir, img))
        else:
            img = cv2.imread(os.path.join(data_dir, img))
            img = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(train_dir, img), img)
        txt_file = img.replace(".jpg", ".txt")
        shutil.copy(os.path.join(data_dir, txt_file), os.path.join(train_dir, txt_file))
        print("copy {} to {}".format(img, train_dir))

file = open(train_file, "w")
for img in train_list:
    path = os.path.join(train_dir, img)
    path = os.path.abspath(path)
    path = path.replace("\\", "/")
    file.write(path + "\n")
file.close()

file = open(val_file, "w")
for img in val_list:
    path = os.path.join(val_dir, img)
    path = os.path.abspath(path)
    path = path.replace("\\", "/")
    file.write(path + "\n")
file.close()

file = open(test_file, "w")
for img in test_list:
    path = os.path.join(test_dir, img)
    path = os.path.abspath(path)
    path = path.replace("\\", "/")
    file.write(path + "\n")
file.close()

shutil.copy(os.path.join(data_dir, "classes.txt"), classes_file)

print("Done!")
