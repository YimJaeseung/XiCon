export CUDA_VISIBLE_DEVICES=$1

nitr=$2


python -u run.py --XiCon   --multiscales 48 --wnorm ReVIN  --lambda 0.001    --d_model 16 --d_ff 16 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/exchange_rate --data_path exchange_rate.csv --model_id XiCon_exp --model XiCon  --data exchange_rate --seq_len 48 --label_len 24 --pred_len 48 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.001 --feature S --omega 0.5 
python -u run.py --XiCon   --multiscales 48 360 --wnorm ReVIN  --lambda 0.01   --d_model 8 --d_ff 16 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/exchange_rate --data_path exchange_rate.csv --model_id XiCon_exp --model XiCon  --data exchange_rate --seq_len 48 --label_len 24 --pred_len 360 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0001 --feature S --omega 0.5
python -u run.py --XiCon   --multiscales 720 --wnorm ReVIN  --lambda 10   --d_model 16 --d_ff 16 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/exchange_rate --data_path exchange_rate.csv --model_id XiCon_exp --model XiCon  --data exchange_rate --seq_len 48 --label_len 24 --pred_len 720 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.00001 --feature S --omega 0.5
python -u run.py --XiCon   --multiscales 48 540 1080 --wnorm ReVIN  --lambda 10   --d_model 8 --d_ff 8 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/exchange_rate --data_path exchange_rate.csv --model_id XiCon_exp --model XiCon  --data exchange_rate --seq_len 48 --label_len 24 --pred_len 1080 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0001 --feature S --omega 0.5


