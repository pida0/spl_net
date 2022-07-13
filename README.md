# SPL_Net
This is the code repository for our paper "SPL-Net: Spatial-Semantic Patch Learning Network for Facial Attribute Recognition with Limited Labeled Data".

Our SPL-Net is to perform FAR with limited labeled data effectively. The
SPL-Net method involves a two-stage learning procedure.
For the first stage, three auxiliary tasks (PRT, PST, and PCT) are jointly developed to exploit the spatial-semantic information on large-scale unlabeled facial data, and thus a
powerful pretrained MSS is obtained. For the second stage,
only a few number of labeled facial data are leveraged to
fine-tune the pretrained MSS and an FAR model is finally
learned. 
## Pytorch
- Python 3.7.11
- torch 1.10.1
- torchvision 0.11.2

## Prepare
- The [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) dataset is required. Random select 30 (or 300) images from CelebA-HQ, and train [BiSeNet-v2](https://arxiv.org/abs/2004.02147) on these images to get a model.
- The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [LFWA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [MAAD](https://github.com/pterhoer/MAAD-Face) datasets are required. Use the trained BiSeNet-v2 model to generate semantic masks of these three datasets for training PST in SPL-Net.

## STAGE 1: train and test pretext task
- modify `config_files/test.yml`
- run `python train_pretext_5b_adv.py`

## STAGE 2: train and test downstream task
- modify `config_files/train_downstream.yml`
- set `PRETRAIN_EPOCH` and `PRETRAIN_PTH` based on the saved model in stage 1
- run `python train_downstream.py`
