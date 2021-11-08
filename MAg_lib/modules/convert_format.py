import json
import os

from numpy import DataSource

def searchFile(key,startPath = '.'):
    if not os.path.isdir(startPath):
        raise ValueError
    l= [os.path.join(startPath,x) for x in os.listdir(startPath)]  #列出所有文件的绝对路径
    #listdir出来的相对路径 不能用于 isfile  abspath只能用在当前目录
    filelist=[x for x in l if os.path.isfile(x) if key in os.path.splitext(os.path.basename(x))[0]] #文件
    #只查找文件名中  不包括后缀 文件路径
    if not hasattr(searchFile,'basePath'):#把函数当成类 添加属性
        searchFile.basePath=startPath #只有第一次调用才会赋值给basePath
    outmap = map(lambda x:os.path.relpath(x,searchFile.basePath),filelist) #转换成相对于初始路径的相对路径

    outlist = list(outmap) 

    dirlist= [x for x in l if os.path.isdir(x)]  #目录
    for dir in dirlist:
        outlist = outlist + searchFile(key,dir)
 
    return outlist

def json_file_to_dict(path):
    #convert json to dict
    with open(path, 'r') as f:
        dict = json.load(fp=f)
    return dict

def dict_to_json_file(dict,path):
    with open(path,'w') as file_obj:
        json.dump(dict,file_obj)
    return

def convert_dataset(json_path,num_features):
    dataset_dict = json_file_to_dict(json_path)
    num_samples = len(dataset_dict)
    X = [[0]*num_features for i in range(dataset_dict)]
    y = []
    count = 0
    for key in dataset_dict:
        feature_tmp = dataset_dict[key]['feature']
        label = dataset_dict[key]['gt_label']
        for j in range(num_features):
            X[count][j] = feature_tmp[j]
        y[count] = label
        count = count + 1
    return X,y