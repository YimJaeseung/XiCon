export CUDA_VISIBLE_DEVICES=$1

nitr=$2
I=336


python -u run.py --XiCon --multiscales 96 --wnorm ReVIN  --lambda 10 --d_model 16 --d_ff 16 --e_layers 4 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id experiment --model XiCon --data ETTm2 --seq_len $I --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.005 --feature S --omega 0.5
python -u run.py --XiCon --multiscales 192 --wnorm ReVIN  --lambda 0.1 --d_model 16 --d_ff 16 --e_layers 4 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id experiment --model XiCon --data ETTm2 --seq_len $I --label_len 48 --pred_len 192 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0001 --feature S --omega 0.5
python -u run.py --XiCon --multiscales 336 --wnorm ReVIN  --lambda 1 --d_model 16 --d_ff 16 --e_layers 4 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id experiment --model XiCon --data ETTm2 --seq_len $I --label_len 48 --pred_len 336 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0005 --feature S --omega 0.5
python -u run.py --XiCon --multiscales 720 --wnorm ReVIN  --lambda 1 --d_model 16 --d_ff 16 --e_layers 4 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTm2.csv --model_id experiment --model XiCon --data ETTm2 --seq_len $I --label_len 48 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0005 --feature S --omega 0.5