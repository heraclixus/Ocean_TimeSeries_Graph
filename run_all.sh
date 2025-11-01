CUDA_VISIBLE_DEVICES=0 nohup bash run_all_nxro.sh --members 100 --rollout_k 1 --epochs 500 --device auto --test --extra_train_nc auto &> log_nxro.txt &
# CUDA_VISIBLE_DEVICES=1 nohup bash run_all_nxro.sh --stochastic --members 100 --rollout_k 1 --epochs 500 --device auto --test --extra_train_nc auto &> log_nxro_stochastic.txt &
# CUDA_VISIBLE_DEVICES=2 nohup bash run_all_nxro_ora5.sh --members 100 --rollout_k 1 --epochs 2000 --device auto --test --topk 3 &> log_nxro_ora5.txt &
# CUDA_VISIBLE_DEVICES=3 nohup bash run_all_nxro_ora5.sh --stochastic --members 100 --rollout_k 1 --epochs 2000 --device auto --test --topk 3 &> log_nxro_ora5_stochastic.txt &
