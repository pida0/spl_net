# spl_net

## train and test pretext task
- modify `config_files/test.yml`
- run `python train_pretext_5b_adv.py`

## train and test downstream task
- modify `config_files/train_downstream.yml`
- set `PRETRAIN_EPOCH` and `PRETRAIN_PTH` based on the saved model in pretext task
- run `python train_downstream.py`
