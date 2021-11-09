import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from MAg_lib.modules.convert_format import json_file_to_dict
from MAg_lib.modules.aggregation import counting, averaging, get_patch_score_list, convert_feature
from sklearn import svm
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score,balanced_accuracy_score

def find_patch(input_path,dataset_path):
    '''
    This is used to find the corresponding patches' pathes from the json 
    of all patches according to the name in the json of the samples.
    input format(sample_name + class_label): {'sample1':0, 'sample12':1, ......}
    output format(sample_name + patch_path): {'sample1':{'class':0, 'patch':[......]}, 'sample12':{'class':0, 'patch':[......]}, ......}
    '''
    sample_dict = json_file_to_dict(input_path)
    patch_dict = json_file_to_dict(dataset_path)
    save_dict = {}
    for key in sample_dict:
        patch_list = patch_dict[key] 
        save_dict[key] = {'class': sample_dict[key], 'patch': patch_list}
    return save_dict

def patch_predict(model,patch_list,prob = True):
    '''
    This is used to get the patch-level prediction results according to a list containing pathes of patches.
    Parameters:
                model: patch-level trained classification models. (e.g. ResNet18, MobileNetV2.....)
                patch_list: a list of pathes. (e.g. [xx/xx/xxx.png, xx/xx/xxx.png, ......])
                prob: get prediction scores or labels(threshold = 0.5)? True: scores; False: labels
    output: a dict like this: {'xx/xx/xxx.png': xxx, 'xx/xx/xxx.png': xxx, ......}
    '''
    save_pred = {}
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
        save_pred[patch_list[i]] = prob_score if prob == True else pred_label
    return save_pred

def get_feature(model,sample_path,hist_num):
    '''
    This is used to get features of each sample from a dict containing sample names and patches' pathes of each sample.
    Parameters:
                model: patch-level trained classification models. (e.g. ResNet18, MobileNetV2.....)
                sample_path: path of the json file from the def find_patch()
                hist_num: the number of features for each sample, that is, how many segments the 0-1 histogram is divided into
    output: a dict saving features like: {'sample1': {'gt_label': 0/1,
                                                      'features': [......] 
                                          'sample1': {'gt_label': 0/1,
                                                      'features': [......] 
                                            ......
                                        }
    '''
    sample_dict = json_file_to_dict(sample_path)
    save_dict = {}
    for key in sample_dict:
        gt_label = sample_dict[key]['class']
        patch_list = sample_dict[key]['patch']
        score_list = get_patch_score_list(model,patch_list,prob=True)
        features = convert_feature(score_list,hist_num)
        save_dict[key] = {'feature':features, 'gt_label':gt_label}
    return save_dict

def find_best_svm(X,y,X_val,y_val,kernal_list,C_list,class_list):
    '''
    This is a function used to find the parameters that make the SVM perform best on the validation set, 
    which is similar to the grid search method. Now we only consider these three parameters(kernal, penalty
    coefficients and class weights), you can also modify this simple function according to your own needs.
    Parameters:
                X,y: training set
                X_val,y_val: validation set
                kernal_list,C_list,class_list: three lists respectively represents kernal, penalty coefficients and class weightsyou want to let it try.
    '''

    parameters = []
    f1_score_list = []
    # print(kernal_list)
    num_kernal = len(kernal_list)
    for a in range(num_kernal):
        for b in range(len(C_list)):
            for c in range(len(class_list)):
                clt = svm.SVC(C = C_list[b],kernel = kernal_list[a],probability = True,class_weight = class_list[c])
                clt.fit(X,y)
                pred_label_val = clt.predict(X_val)
                f1 = f1_score(y_val,pred_label_val)
                f1_score_list.append(f1)
                parameters.append({'kernal':kernal_list[a],'C':C_list[b],'class_weight':class_list[c]})
                # print('parameters',{'kernal':kernal_list[a],'C':C_list[b],'class_weight':class_list[c]},'f1',f1)
    best_parameters = parameters[f1_score_list.index(max(f1_score_list))]
    # best_svm = svm.SVC(C = best_parameters['C'],kernel = best_parameters['kernal'],probability = True,class_weight = best_parameters['class_weight'])
    return best_parameters


def patient_predict(model,sample_path,method,hist_num,svm):
    '''
    This is used to make the patient_level prediction from the json dict containing sample names and patches' pathes of each sample.
    Parameters:
                model: patch-level trained classification models. (e.g. ResNet18, MobileNetV2.....)
                sample_path: path of the json file from the def find_patch()
                method: select the aggregation method ('counting', 'averaging', 'MAg')
                histnum: represents the number of dimensions and is required in 'MAg' method only
                svm: represents the SVM in MAg and is also required in 'MAg' method only
    output: a dict saving the results like: {'sample1': {'pred_score': xxx, 'pred_label': x, 'gt_label': x}
                                             'sample2': {'pred_score': xxx, 'pred_label': x, 'gt_label': x} 
                                            }
    '''
    sample_dict = json_file_to_dict(sample_path)
    save_dict = {}
    for key in sample_dict:
        gt_label = sample_dict[key]['class']
        patch_list = sample_dict[key]['patch']
        score_list = get_patch_score_list(model,patch_list,prob=True)
        if method == 'counting':
            patient_score = counting(score_list)
            patient_label = 1 if patient_score >= 0.5 else 0
        elif method == 'averaging':
            patient_score = averaging(score_list)
            patient_label = 1 if patient_score >= 0.5 else 0
        elif method == 'MAg':
            features = convert_feature(score_list,hist_num)
            patient_score = svm.predict(features)
            patient_label = 1 if patient_score >= 0.5 else 0
        save_dict[key] = {'pred_score':patient_score, 'pred_label':patient_label, 'gt_label':gt_label}
    return save_dict

def evaluate(X,y,svm):
    '''
    This is used to evaluate the performance of SVM on a set(X,y)
    output: a dict showing the performance of the SVM like: {'pred_label':[...],'pred_score':[...],'f1_score':xxx,'BACC':xxx,'auc':xxx}
    '''
    output_label = svm.predict(X)
    prob_list = svm.predict_proba(X)
    output_prob = []
    for i in range(len(prob_list)):
        output_prob.append(prob_list[i][1])
    f1 = f1_score(y,output_label)
    BACC = balanced_accuracy_score(y,output_label)
    auc_score = roc_auc_score(y,output_prob)
    return {'pred_label':output_label,'pred_score':output_prob,'f1_score':f1,'BACC':BACC,'auc':auc_score}