# XiCon

This is an official repository for the paper, titled [Contrastive learning for long term time series forecasting with ξ- correlation].


## Environment setup
```
python == 3.8
torch == 1.7.1
numpy == 1.23.5
pandas
statsmodels
scikit-learn
einops
sympy
numba
```

## Run with Command Line 

"""
python -u run.py --XiCon --multiscales 96  --wnorm ReVIN  --lambda 1.0 --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id ICLR24_CRV --model XiCon --data ETTh1 --seq_len 336 --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr 5 --batch_size 64 --learning_rate 0.01 --feature S --omega 0.3

"""

## Run with Scripts
sh ./scripts/XiCon_{ETTh1|ETTh2|ETTm1|ETTm2|Electricity|Traffic|Weather|Excange|Illness} {CUDA_VISBLE_DEVICES} {# OF RUNS}

Examples
```
$pwd
/home/user/XiCon

$sh ./scripts/XiCon_ETTh2.sh 0 5 
$sh ./scripts/XiCon_Traffic.sh 0 5
```

## Reproducibility
Files ending in a number (e.g., XiCon_Elec_s_revin_e-3) are for lambda hyperparameter tuning.

Files ending in aa(omega=0.99 ; almost AutoCon), half(omega=0.5), or ax(omega=0.01; almost XiCon) (e.g., XiCon_Electricity_s_revin_aa) are for omega hyperparameter tuning.

## Citations
