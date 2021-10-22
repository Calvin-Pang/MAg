# MAg: a simple learning-based patient-level aggregation method for detecting microsatellite instability from whole-slide images
The prediction of microsatellite instability (MSI) and microsatellite stability (MSS) is essential in predicting both the treatment response and prognosis of gastrointestinal cancer. In clinical practice, a universal MSI testing is recommended, but the accessibility of such a test is limited. Thus, a more cost-efficient and broadly accessible tool is desired to cover the traditionally untested patients. In the past few years, deep-learning-based algorithms have been proposed to predict MSI directly from haematoxylin and eosin (H&E)-stained whole-slide images (WSIs). Such algorithms can be summarized as (1) patch-level MSI/MSS prediction, and (2) patient-level aggregation. Compared with the advanced deep learning approaches that have been employed for the first stage, only the naïve first-order statistics (e.g., averaging and counting) were employed in the second stage. In this paper, we propose a simple yet broadly generalizable patient-level MSI aggregation (MAg) method to effectively integrate the precious patch-level information. Briefly, the entire probabilistic distribution in the first stage is modeled as histogram-based features to be fused as the final outcome with machine learning (e.g., SVM). The proposed MAg method can be easily used in a plug-and-play manner, which has been evaluated upon five broadly used deep neural networks: ResNet, MobileNetV2, EfficientNet, Dpn and ResNext. From the results, the proposed MAg method consistently improves the accuracy of patient-level aggregation for two publicly available datasets. It is our hope that the proposed method could potentially leverage the low-cost H&E based MSI detection method.
The comparison of our method and the two common used method (counting and averaging) is shown below:

<div align=center><img src="https://user-images.githubusercontent.com/72646258/138407969-c2e5ce61-4957-487f-98b9-a249042fcdf4.png" height="500"/><br/></div>
The proposed method is shown in figure below:

<div align=center><img src="https://user-images.githubusercontent.com/72646258/138407814-e2888b56-878c-4ea3-998a-4ffa511e1c95.png" height="300"/><br/></div>

# File structure
![image](https://user-images.githubusercontent.com/72646258/138487068-c231137c-0da2-4850-8fe8-104faf5d5cc8.png)

# Dataset prepare
1.The whole patch-level datasets can be downloaded from https://zenodo.org/record/2530835#.YXIlO5pBw2z. Each patch in this folder belongs to a patient, and the file name of the patch can be used to get to which patient it belongs. For example, the patch **blk-AAAFIYHTSVIE-TCGA-G4-6309-01Z-00-DX1.png** belongs to the patient **TCGA-G4-6309**.

2.We have split the CRC_DX and STAD datasets into training set, validation set and testing set in the patient-level. So after downloading them from the link, please split the dataset according to the patient name we list in the **name_patient** file. 

3.Certainly, if you want to change the way of splitting the data set, you can also split the data set yourself. For your reference, you can use the code in link https://github.com/jnkather/MSIfromHE/blob/master/step_05_split_train_test.m to do this split.
# How to use MAg?
The code of our method is in the **demo** file. Follow the steps below, you can easily use MAg to complete training and prediction.

1.Firstly, please use **1.patch-level classification training.ipynb** to do patch-level training and get classification models. The Timm library is such a creative invention that it can help you easily complete this training process. For example, if you want to use ResNet18 in this stage, just use the code below after entering the working file:

```
import timm
!python train.py path_to_your_dataset -d ImageFolder --drop 0.25 --train-split train --val-split validation --pretrained --model resnet18 --num-classes 2 --opt adam --lr 1e-6 --hflip 0.5 --epochs 40 -b 32 --output path_to_your_model
```
The script **train.py** and other scripts useful in Timm can be obtained from this link: https://github.com/rwightman/pytorch-image-models 
Also, here are some very helpful links that teachs you how to use Timm: https://fastai.github.io/timmdocs/ and https://rwightman.github.io/pytorch-image-models/

2.Secondly, after using the above process to obtain the classification model in the patch-level, you can use **2.0.patch2image_counting.ipynb** to make the patch-level prediction. In this process, just follow the operation of the code in the notebook and you can get patch-level probabilities and histogram-based features. 

**NOTE**: in our experiments, we use xlsx format files to save predicted probability scores and other data. You can also freely modify the code to use other formats such as json format files to save the results, but it may make the code more complicated.

3.After getting the patch-level model and patient-level histogram-based features from processes above, you can now train the patient-level classification models. Here we provide two different methods to complete this training, which means you can train it in an SVM with **2.1.patient-level MAg-SVM_histogram.ipynb** or in a two-layer fully-connected neural network with **2.2.patient-level MAg-network.ipynb**. If your dataset is not very large, we suggest you using the **2.1** while you had better use **2.2** if you have a hugh dataset(e.g, a dataset which contains 100000 patients).

4.Both the **2.1** and the **2.2** contain code that can do a simple testing process. After getting the patient-level classification model, just continue following the code in these two notebooks and you will get the final result (e.g, F1 score, BACC and AUC).

**NOTE**: the patient-level training also require you to follow the split you did before, so please remember to save the patient-level histogram-based features in xlsx files like train.xlsx, validation.xlsx and test.xlsx

5.In the **demo** file, we also provide some notebooks whose file names start with 0. These demos are used by us in our experiment. Although they are not directly related to the MAg process, we think they may be able to help you in your own experiment. Their roles are different. For example, **0.3.confusion_matrix.ipynb** can help you calculate a patient-level confusion matrix. The role of each demo can be viewed at the beginning of their code.

# Experiment and results
The experiments were performed on a Google Colab workstation with a NVIDIA Tesla P100 GPU. In stage I, five prevalent approaches have been used to be the baseline feature extractors, including ResNet, MobileNetV2, EfficientNet, Dpn, and ResNext models. And in stage II, we mainly use SVM to complete it. Moreover, to assess the generalizability, the experiments above were done in both the CRC dataset and the STAD dataset.
Below is the results of our experiments and comparison between MAg and two commonly used methods (counting and averaging):

<img src="https://user-images.githubusercontent.com/72646258/138465280-e289b796-d3db-47c3-9c79-a2b355fc156f.png" height="220"/><img src="https://user-images.githubusercontent.com/72646258/138465330-1668c95f-b545-4cdb-93c3-a219e7d8be5c.png" height="220"/><br/>
