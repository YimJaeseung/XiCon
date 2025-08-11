export CUDA_VISIBLE_DEVICES=$1

nitr=$2


python -u run.py --XiCon  --multiscales 336 --wnorm ReVIN  --lambda 0.1  --d_model 4 --d_ff 4 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/weather --data_path weather.csv --model_id XiCon_exp --model XiCon --data weather --seq_len 336 --label_len 48 --pred_len 96 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0003 --feature S --omega 0.5
python -u run.py --XiCon  --multiscales 720 --wnorm ReVIN  --lambda 10  --d_model 4 --d_ff 4 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/weather --data_path weather.csv --model_id XiCon_exp --model XiCon --data weather --seq_len 336 --label_len 48 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0001 --feature S  --omega 0.5
python -u run.py --XiCon  --multiscales 336 --wnorm ReVIN  --lambda 1  --d_model 8 --d_ff 8 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/weather --data_path weather.csv --model_id XiCon_exp --model XiCon --data weather --seq_len 336 --label_len 48 --pred_len 1440 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0001  --feature S  --omega 0.5
python -u run.py --XiCon  --multiscales 720 --wnorm ReVIN  --lambda 10  --d_model 8 --d_ff 8 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/weather --data_path weather.csv --model_id XiCon_exp --model XiCon --data weather --seq_len 336 --label_len 48 --pred_len 2160 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 128 --learning_rate 0.0001 --feature S  --omega 0.5

