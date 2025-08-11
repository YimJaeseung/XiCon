export CUDA_VISIBLE_DEVICES=$1

nitr=$2
I=336

python -u run.py --XiCon --multiscales 96  --wnorm ReVIN  --lambda 10 --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id experiment --model XiCon --data ETTh1 --seq_len $I --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.01 --feature S --omega 0.99
python -u run.py --XiCon --multiscales 720  --wnorm ReVIN  --lambda 1  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id experiment --model XiCon --data ETTh1 --seq_len $I --label_len 48 --pred_len 196 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S --omega 0.99
python -u run.py --XiCon --multiscales 1440  --wnorm ReVIN  --lambda 1 --d_model 16 --d_ff 16 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id experiment --model XiCon --data ETTh1 --seq_len $I --label_len 48 --pred_len 336 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S --omega 0.99
python -u run.py --XiCon --multiscales 1440  --wnorm ReVIN  --lambda 0.001 --d_model 16 --d_ff 32 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh1.csv --model_id experiment --model XiCon --data ETTh1 --seq_len $I --label_len 48 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.005 --feature S --omega 0.5