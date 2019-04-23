# A re-implementation of Recognition Adversary Network (RAN)


A PyTorch re-implementation of Weakly Supervised Facial Action Unit Recognition through Adversarial Training

## Generate Pseudo AU vectors

```sh
python tools/gen_pseudo_au.py
```

## Preprocess CK+ dataset 

```sh
python tools/preprocess_ckplus.py
```

## Train 

```sh
python main.py --data_root datasets/CKPlus --which_model_netR resnet18 --backend_pretrain --gpu_ids 0 --gan_type wgan-gp --load_size 250 --final_size 224 --visdom_env resnet18_wgan_fold1 --train_csv train_ids_1.csv --test_csv test_ids_1.csv
```

## Test 

```sh
python main.py --mode test --data_root datasets/CKPlus --gpu_ids 0 --ckpt_dir ckpts/CKPlus/resnet18/fold_1/190423_105211 --load_epoch 300 --which_model_netR resnet18 --load_size 250 --final_size 224 --test_csv test_ids_1.csv
```

