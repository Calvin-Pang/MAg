import numpy as np
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_patch_score_list(model,patch_list,prob = True):
    '''
    This is used to make the patch_level prediction from a list of pathes of patches.
    output: a list of prediction results
    '''
    save_pred = []
    for i in range(len(patch_list)):
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        image = Image.open(patch_list[i]).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0).numpy()
        prob_score = probabilities[0]
        pred_label = 1 if prob_score >= 0.5 else 0
        save_pred.append(prob_score if prob == True else pred_label)
    return save_pred

def convert_feature(score_list,hist_num):
    '''
    This is used to convert the score list to the feature list
    Parameters:
                score_list: a list containing the scores of all patches of a sample
                hist_num: number of features
    '''
    counts,bins = np.histogram(score_list,bins = hist_num,range = (0,1))
    counts_norm = counts / len(score_list)
    return counts_norm


def counting(score_list,threshold = 0.5):
    '''
    This is used to convert score_list to the sample score with the counting method.
    '''
    num = 0
    for i in range(len(score_list)):
        if score_list[i] >= 0.5:
            num  = num + 1
    return num / len(score_list)


def averaging(score_list):
    '''
    This is used to convert score_list to the sample score with the averaging method.
    '''
    return sum(score_list) / len(score_list)


