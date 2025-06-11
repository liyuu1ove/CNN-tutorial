import xml.etree.ElementTree as ET

import os
from os import listdir, getcwd

classes = ["data1"]

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_name,labelPath):

    out_file = open(os.path.join(labelPath+'TXT',image_name[:-3] + 'txt'), 'w')  # 转换后的txt文件存放路径
    in_file = open(os.path.join(labelPath,image_name[:-3] + 'xml'))  # xml文件路径
    
    xml_text = in_file.read()
    root = ET.fromstring(xml_text)
    in_file.close()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if __name__ == '__main__':
    imgPath='C:/Users/86183/Desktop/ultralytics/datasets/test/test/img'
    labelPath='C:/Users/86183/Desktop/ultralytics/datasets/test/test/data'
    if not os.path.isdir(labelPath+'_yolo'):
        os.mkdir(labelPath+'_yolo')
    for image_path in os.listdir(imgPath):
        print(image_path)
        convert_annotation(image_path,labelPath)
