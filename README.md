# MAg: a simple learning-based patient-level aggregation method for detecting microsatellite instability from whole-slide images
The prediction of microsatellite instability (MSI) and microsatellite stability (MSS) is essential in predicting both the treatment response and prognosis of gastrointestinal cancer. In clinical practice, a universal MSI testing is recommended, but the accessibility of such a test is limited. Thus, a more cost-efficient and broadly accessible tool is desired to cover the traditionally untested patients. In the past few years, deep-learning-based algorithms have been proposed to predict MSI directly from haematoxylin and eosin (H&E)-stained whole-slide images (WSIs). Such algorithms can be summarized as (1) patch-level MSI/MSS prediction, and (2) patient-level aggregation. Compared with the advanced deep learning approaches that have been employed for the first stage, only the naïve first-order statistics (e.g., averaging and counting) were employed in the second stage. In this paper, we propose a simple yet broadly generalizable patient-level MSI aggregation (MAg) method to effectively integrate the precious patch-level information. Briefly, the entire probabilistic distribution in the first stage is modeled as histogram-based features to be fused as the final outcome with machine learning (e.g., SVM). The proposed MAg method can be easily used in a plug-and-play manner, which has been evaluated upon five broadly used deep neural networks: ResNet, MobileNetV2, EfficientNet, Dpn and ResNext. From the results, the proposed MAg method consistently improves the accuracy of patient-level aggregation for two publicly available datasets. It is our hope that the proposed method could potentially leverage the low-cost H&E based MSI detection method.
The comparison of our method and the two common used method (counting and averaging) is shown below:
![fig1-4 0](https://user-images.githubusercontent.com/72646258/138407969-c2e5ce61-4957-487f-98b9-a249042fcdf4.png)
The proposed method is shown in figure below:
![fig2-4 0](https://user-images.githubusercontent.com/72646258/138407814-e2888b56-878c-4ea3-998a-4ffa511e1c95.png)
# Dataset prepare
1.The whole patch-level datasets can be downloaded from https://zenodo.org/record/2530835#.YXIlO5pBw2z. Each patch in this folder belongs to a patient, and the file name of the patch can be used to get to which patient it belongs. For example, the patch **blk-AAAFIYHTSVIE-TCGA-G4-6309-01Z-00-DX1.png** belongs to the patient **TCGA-G4-6309**.
2.We have split the CRC_DX and STAD datasets into training set, validation set and testing set in the patient-level. So after downloading them from the link, please split the dataset according to the patient name we list in the **name_patient** file. 
# How to use MAg?
The code of our method is in the demo file. Firstly, please use 1.patch-level classification training.ipynb to get patch-level classification model. Secondly, use 2.0.patch2image_counting.ipynb to get patch-level probabilities and histogram-based features. Finally, you can use 2.1 or 2.2 to train MAg model in the patient-level and test it.
