import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from MAg_lib.modules.convert_format import json_file_to_dict
from MAg_lib.modules.aggregation import counting, averaging, get_patch_score_list, convert_feature
from sklearn import svm
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score,balanced_accuracy_score

def find_patch(input_path,dataset_path):
    sample_dict = json_file_to_dict(input_path)
    patch_dict = json_file_to_dict(dataset_path)
    save_dict = {}
    for key in sample_dict:
        patch_list = patch_dict[key] 
        save_dict[key] = {'class': sample_dict[key], 'patch': patch_list}
    return save_dict

def patch_predict(model,patch_list,prob = True):
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
    # hist_num: represents the number of dimensions and is required in 'MAg' method only
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
    output_label = svm.predict(X)
    prob_list = svm.predict_proba(X)
    output_prob = []
    for i in range(len(prob_list)):
        output_prob.append(prob_list[i][1])
    f1 = f1_score(y,output_label)
    BACC = balanced_accuracy_score(y,output_label)
    auc_score = roc_auc_score(y,output_prob)
    return {'pred_label':output_label,'pred_score':output_prob,'f1_score':f1,'BACC':BACC,'auc':auc_score}