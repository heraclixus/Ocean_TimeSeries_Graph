# CUDA_VISIBLE_DEVICES=6 nohup python run_graph_models.py --model_name=graphode --use_normalization --use_loss_weights --use_region_only --use_region_data --batch_size=128 --gnn_latent_dim=32 --graph_encoder=gcn --hidden_size=64 --learning_rate=0.00001 --ode_encoder_decoder &> log_graphode_config1.txt &
# CUDA_VISIBLE_DEVICES=7 nohup python run_graph_models.py --model_name=graphode --use_normalization --use_loss_weights --use_region_only --use_region_data --batch_size=32 --gnn_latent_dim=32 --graph_encoder=gcn --hidden_size=32 --learning_rate=0.0001 --use_periodic_activation &> log_graphode_config2.txt &
CUDA_VISIBLE_DEVICES=4 nohup python run_graph_models.py --model_name=agcrn --use_normalization --use_loss_weights --use_region_only --use_region_data --batch_size=32 --hidden_size=128 --cheb_k=2 --learning_rate=0.0001 --num_layers=3 --rnn_units=32 &> log_agcrn.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_graph_models.py --model_name=fgnn --use_normalization --use_loss_weights --use_region_only --use_region_data --batch_size=32 --hidden_size=128 --embed_dim=32 --learning_rate=0.0001 --rnn_units=128 &> log_fgnn.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_graph_models.py --model_name=mtgnn --use_normalization --use_loss_weights --use_region_only --use_region_data --batch_size=16 --hidden_size=64 --learning_rate=0.0001 &> log_mtgnn.txt &

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
# model=stemgnn
# window=12
# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --model_name=$model --use_normalization --use_loss_weights --batch_size=64 --window=$window &> log_debug_stemgnn.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --use_cosine --n_pcs=20 --model_name=$model --use_normalization --use_loss_weights --batch_size=96 --window=$window &> log_debug_stemgnn_cosine.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --use_cosine --use_warmup --n_pcs=20 --model_name=$model --use_normalization --use_loss_weights --batch_size=96 --window=$window &> log_debug_stemgnn_cosine_warmup.txt & 



# for model in {agcrn,}; do
#     for window in {5,6}; do
#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}_weighted_32_${window}.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}_weighted_64_${window}.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}_weighted_96_${window}.txt &

#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}_weighted_32_${window}.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}_weighted_64_${window}.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}_weighted_96_${window}.txt &        

#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}_weighted_32_${window}_cosine.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}_weighted_64_${window}_cosine.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}_weighted_96_${window}_cosine.txt &

#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}_weighted_32_${window}_cosine.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}_weighted_64_${window}_cosine.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}_weighted_96_${window}_cosine.txt &

#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}_weighted_32_${window}_cosine_warmup.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}_weighted_64_${window}_cosine_warmup.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}_weighted_96_${window}_cosine_warmup.txt &

#         CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}_weighted_32_${window}_cosine_warmup.txt &
#         CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}_weighted_64_${window}_cosine_warmup.txt &
#         CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}_weighted_96_${window}_cosine_warmup.txt &

#     done
# done


# for model in {agcrn,}; do
#     for window in {5,6}; do
#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_32_${window}_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_64_${window}_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_96_${window}_sin-cos.txt &

#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_32_${window}_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_64_${window}_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_96_${window}_sin-cos.txt &        

#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_32_${window}_cosine_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_64_${window}_cosine_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_96_${window}_cosine_sin-cos.txt &

#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_32_${window}_cosine_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_64_${window}_cosine_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_96_${window}_cosine_sin-cos.txt &

#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_32_${window}_cosine_warmup_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_64_${window}_cosine_warmup_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_96_${window}_cosine_warmup_sin-cos.txt &

#         CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_32_${window}_cosine_warmup_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_64_${window}_cosine_warmup_sin-cos.txt &
#         CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_96_${window}_cosine_warmup_sin-cos.txt &

#     done
# done
# model=agcrn
# CUDA_VISIBLE_DEVICES=1 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=5 --use_cosine --add_sin_cos &> logs_report_5/log_${model}_weighted_64_5_cosine_sin-cos.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=6 --use_cosine --use_warmup --add_sin_cos &> logs_report/log_${model}_weighted_64_6_cosine_warmup_sin-cos.txt &
# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=5 --use_cosine --add_sin_cos &> logs_report/log_${model}_weighted_64_5_cosine_sin-cos.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=6 --add_sin_cos &> logs_report/log_${model}_weighted_64_6_sin-cos.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=6 --use_cosine &> logs_report/log_${model}_weighted_96_6_cosine.txt &


# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --epochs=2 --model_name=fgnn --use_normalization --use_loss_weights --use_cosine --add_sin_cos &> log_fgnn_debug.txt &

# CUDA_VISIBLE_DEVICES=7 nohup python run_pygtemporal_models.py --model_name=stemgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_stemgnn_weighted_${batch_size}_${window}.txt &
# CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --model_name=fgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_fgnn_weighted_${batch_size}_${window}.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --model_name=mtgnn --use_normalization --use_loss_weights --batch_size=$batch_size --window=$window &> logs_report/log_mtgnn_weighted_${batch_size}_${window}.txt & 

# stemgnn done done
# fgnn done done 
# mtgnn done done
# agcrn done done
# wavenet fail 
# model=agcrn
# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=18 &> logs_report_5/log_${model}_weighted_64_18_cosine_warmup.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=12 &> logs_report_5/log_${model}_weighted_96_12_cosine_warmup.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=12 &> logs_report/log_${model}_weighted_96_12_cosine_warmup.txt &

# CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=36 &> logs_report/log_${model}_weighted_64_36_cosine.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=12 &> logs_report/log_${model}_weighted_32_12_cosine.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=24 &> logs_report/log_${model}_weighted_32_24_cosine.txt &
# CUDA_VISIBLE_DEVICES=6 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=36 &> logs_report/log_${model}_weighted_32_36.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=fgnn --batch_size=32 --window=12 &> logs_report/log_fgnn_weighted_32_12.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=mtgnn --batch_size=64 --window=18 &> logs_report/log_mtgnn_weighted_64_18.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pygtemporal_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=stemgnn --batch_size=32 --window=18 &> logs_report_5/log_stemgnn_weighted_32_18.txt &


# CUDA_VISIBLE_DEVICES=3 nohup python run_pygtemporal_models.py --n_pcs=5 --use_cosine --use_warmup --use_loss_weights --use_normalization --model_name=agcrn --batch_size=32 --window=12 &> logs_report/log_agcrn_weighted_32_12_cosine_warmup.txt &
