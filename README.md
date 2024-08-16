# Multi-level Negative Contrastive Learning for Text to Point Cloud Localization

---

This repository is the official implementation of our model RoLo

![](C:\Users\ZhuanZ\OneDrive\Desktop\论文\AAAI_2025\论文插图(10).png)

# Abstract

---

Language-based localization is a crucial task in robotics and computer vision, enabling robots to understand spatial positions through language. Recent method rely on contrastive learning to establish correspondences between global features of text and point cloud. However, the inherent ambiguity of textual descriptions makes it almost impossible to convey geometric information accurately, forcing alignment them in the feature space may compromise the expressiveness of the point clouds. Unlike pervious methods, this paper propose using language as a filter to distinguish dissimilar
locations. To this end, we propose a Robust framework of multi-level negative contrastive learning for language-based Localization, named NegLoc, to fully leverage the descriptive power of language for spatial localization. NegLoc learns the mismatched factors by minimizing the similarity of different
locations at diffent levels, including instance-level, relation-level, and global-level, respectively. Results show that NegLoc achieves a 56.3% increase in Top-1 retrieval recall rate on the KITTI360Pose benchmark. Moreover, without modifying the stage of fine localization, our model improves the
overall localization accuracy within 5 meters by 45.9%. 

# Create Environment

---

Create the environment using the following command.

```
mkdir MSP
cd MSP

conda create -n msp python=3.10
conda activate msp

# Install the according versions of torch and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```



# Dependencies 

---



Our core dependencies are

```
easydict==1.13
h5py==3.11.0
matplotlib==3.8.3
matplotlib-inline==0.1.6
nltk==3.8.1
numpy==1.26.4
open3d==0.18.0
opencv-python==4.9.0.80
pandas==2.2.1
pillow==10.2.0
plyfile==1.0.3
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tokenizers==0.19.1
torch-cluster==1.6.0+pt112cu113
torch_geometric==2.5.0
torch-scatter==2.1.0+pt112cu113
torch-sparse==0.6.16+pt112cu113
torch-spline-conv==1.2.1+pt112cu113
torch-tb-profiler==0.4.3
torchaudio==0.12.1
torchvision==0.13.1
tqdm==4.65.0
transformers==4.40.0
transforms3d==0.4.1

```

with as an optional package to visualize the point clouds. Please pay close attention to the combined installation of CUDA, PyTorch and PyTorch Geometric as their versions are inter-connected and also depend on your available GPU. It might be required to install these packages by hand with specific versions instead of using the Pip-defaults.`pptk`



# Dataset

---

We use the publicly available dataset KITTI360Pose as our baseline. You can download the KITTI360Pose dataset from [here](https://cvg.cit.tum.de/webshare/g/text2pose/). Once downloaded, the dataset folder should display as follow:

```html
data
└── KITTI360Pose
    └── k360_30-10_scG_pd10_pc4_spY_all
        ├── cells
        ├── direction
        ├── poses
        ├── street_centers
        └── visloc
```



# Pretrained model

---

We use Google's open-source large language model (LLM) Flan-T5 as our text encoder, which can be obtained from [here]([google/flan-t5-large · Hugging Face](https://huggingface.co/google/flan-t5-large)), it can also be replaced with any other language model. For fine-localization, we use the regressor of [Text2Loc](We use the regressor of Text2Loc for fine localization, and the pretrained model has already been provided in the folder.) for fine localization,  the pretrained model has already been provided in the folder.



After completing the above steps, the basic directory structure should be like:

```
MSP
 ├── checkpoints
      ├── coarse.pth
      ├── fine.pth
      └── pointnet_acc0.86_lr1_p256_model.pth
 ├── data
      └── KITTI360Pose
            └── k360_30-10_scG_pd10_pc4_spY_all
                ├── cells
                ├── direction
                ├── poses
                ├── street_centers
                └── visloc
 ├── dataloading
      └── .....
 ├── datapreparation
      └── .....
 ├── evalution
      └── .....
 ├── models
      └── .....
 ├── t5-large
      └── .....
 ├── training
      └── .....
```



# Train

---

After setting up the dependencies and dataset, our models can be trained using the following commands:



#### coarse-localization

```
python -m training.coarse  \
  --batch_size 64  \
  --coarse_embed_dim 256  \
  --shuffle  \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```



#### fine-localization

```
python -m training.fine 
  --batch_size 32 \ 
  --fine_embed_dim 128 \ 
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_FINE \
```



# Evaluation

---

### Evaluation only coarse-localization on Val Dataset

```
python -m evaluation.coarse
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```



### Evaluation only coarse-localization on Test Dataset

```
python -m evaluation.coarse 
	--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
```



### Evaluation whole pipeline on Val Dataset

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} \
```



### Evaluation whole pipeline on Test Dataset

```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```

