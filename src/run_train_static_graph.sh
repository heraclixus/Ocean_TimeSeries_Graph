# CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --model_name=wavenet --use_normalization &> log_wavenet.txt &
# CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=stemgnn --use_normalization &> log_stemgnn.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pygtemporal_models.py --model_name=agcrn --use_normalization &> log_agcrn.txt &
# CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=fgnn --use_normalization &> log_fgnn.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pygtemporal_models.py --model_name=mtgnn --use_normalization &> log_mtgnn.txt & 

# CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=stemgnn &> log_stemgnn_nonorm.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pygtemporal_models.py --model_name=agcrn &> log_agcrn_nonorm.txt &
# CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=fgnn &> log_fgnn_nonorm.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pygtemporal_models.py --model_name=mtgnn &> log_mtgnn_nonorm.txt & 

# batch_size=64
# window=12
# mkdir -p logs_report_5

# for batch_size ibn {128,}; do
#     for window in {36,}; do
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=stemgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report_5/log_stemgnn_weighted_${batch_size}_${window}.txt &
#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=agcrn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report_5/log_agcrn_weighted_${batch_size}_${window}.txt &
#         CUDA_VISIBLE_DEVICES=7 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=mtgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report_5/log_mtgnn_weighted_${batch_size}_${window}.txt & 
#         CUDA_VISIBLE_DEVICES=7 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=fgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report_5/log_fgnn_weighted_${batch_size}_${window}.txt & 
#     done
# done
model=agcrn

for window in {4,8,12,18,24,36}; do
    # CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=$model --use_normalization --use_loss_weights --batch_size=32 --window=$window &> logs_report_5/log_${model}_weighted_32_${window}.txt &
    # CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=$model --use_normalization --use_loss_weights --batch_size=64 --window=$window &> logs_report_5/log_${model}_weighted_64_${window}.txt &
    # CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=$model --use_normalization --use_loss_weights --batch_size=96 --window=$window &> logs_report_5/log_${model}_weighted_96_${window}.txt &
    # CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --model_name=$model --use_normalization --use_loss_weights --batch_size=128 --window=$window &> logs_report_5/log_${model}_weighted_128_${window}.txt &
done
# CUDA_VISIBLE_DEVICES=7 nohup python run_pygtemporal_models.py --model_name=agcrn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_agcrn_weighted_${batch_size}_${window}.txt &

# CUDA_VISIBLE_DEVICES=7 nohup python run_pygtemporal_models.py --model_name=stemgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_stemgnn_weighted_${batch_size}_${window}.txt &
# CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --model_name=fgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_fgnn_weighted_${batch_size}_${window}.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --model_name=mtgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_mtgnn_weighted_${batch_size}_${window}.txt & 

# stemgnn done done
# fgnn done done 
# mtgnn done done
# agcrn done done
# wavenet fail 