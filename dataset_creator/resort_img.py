import os

def file_rename():
    num = 0
    source_dir = "./raw"
    target_dir = "./dataset/data"
    filelist = os.listdir(source_dir)
    for file in filelist:
        Olddir = os.path.join(source_dir, file)    #原来的文件路径
        if os.path.isdir(Olddir):       #如果是文件夹则跳过
                continue
        filename = 'img-'     
        filetype = '.jpg'       
        Newdir = os.path.join(target_dir, filename + str(num) + filetype)   #新的文件路径
        os.rename(Olddir, Newdir)   
        num = num + 1
    return True

if __name__ == '__main__':
    file_rename()