# [Precision-Enhanced Image Attribute Prediction Model](https://github.com/ThroneHU/EapLda)
---
## C. Hu, J. Miao, Z. Su, X. Shi, Q. Chen, X. Luo 
---
-- This project is an oral presentation at IEEE TrustCom-BigDataSE-ICESS 2017.

[(Project Page)](https://ieeexplore.ieee.org/abstract/document/8029527?casa_token=nToodSUyVFgAAAAA:b9H4Ih4p1FDqDLZZYDlbyZGo1iBG9644Pu-Cl-DNeQ8NNEzNwimfEXnvP0Pb-rcaCKwgxlMu) [(PDF)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8029527) 

<!--
[(Slides)](https://sites.google.com/view/tyronehu/research/pointgrasp) [(Video)](https://sites.google.com/view/tyronehu/research/pointgrasp)
-->

![image](https://github.com/ThroneHU/EapLda/blob/main/figs/fig1.jpg)

### Abstract

High-precision attribute prediction is a challeng-ing issue due to the complex object and scene variations. Targeting on enhancing attribute prediction precision, we propose an Enhanced Attribute Prediction-Latent Dirichlet Allocation (EAP-LDA) model to address this issue. EAP-LDA model enhances the attribute prediction precision in two steps: classiﬁcation adaptation and prediction enhancement. In classiﬁcation adaptation, we transfer image low-level features to mid-level features (attributes) by the SVM classiﬁers, which are trained using the low-level features extracted from images. In prediction enhancement, we ﬁrst exploit its advantages in extracting and analyzing the topic information between image samples and attributes by the LDA topic model. We then use a strategy to search the nearest neighbor image collection from test datasets by KNN. Finally, we evaluate the accuracy on HAT datasets and demonstrate signiﬁcant improvement over the baseline algorithm.

<!--
### Installation

1. Install requirements:
```python
pip3 install -r requirements.txt
```

2. Clone the repository using the command:
```python
git clone https://github.com/ehsanik/touchTorch
cd touchTorch
```

### PointNet++

1. Download the PointNet++ from [here](https://github.com/charlesq34/pointnet2).

2. Move the train and Inference scripts to the solver directory.

### Data Preparation

1. Annotate point cloud data using *semantic-segmentation-editor* and collect **.pcd* labels.

2. Generate **.npy* data for `train/val/` by modifying mode.
```python
cd utils
python collect_biorob_3data.py --mode train
```

Processed data will be saved in `datasets/sub_ycb/train`.

### Run
```python
python train_complex.py
python test_e2e.py
```
-->

### Reference By
[HAT dataset](https://grvsharma.com/datasets/hat/)

### Citation

If you find this project useful in your research, please consider citing:
```
@inproceedings{hu2017precision,
  title={Precision-enhanced image attribute prediction model},
  author={Hu, Chen and Miao, Jiaxin and Su, Zhuo and Shi, Xiaohong and Chen, Qiang and Luo, Xiaonan},
  booktitle={2017 IEEE Trustcom/BigDataSE/ICESS},
  pages={866--872},
  year={2017},
  organization={IEEE}
}
```
