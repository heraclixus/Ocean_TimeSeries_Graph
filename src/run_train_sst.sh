
# # LGODE without GAT 

# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=lgode_sst_0 --fourier_coeff=0 &> log_lgode_sst_f0.txt &  
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=lgode_sst_200 --fourier_coeff=200 &> log_lgode_sst_f200.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=lgode_sst_500 --fourier_coeff=500 &> log_lgode_sst_f500.txt &

# # LGODE with GAT

# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --use_gat --save_name=lgode_gat_sst_0 --fourier_coeff=0 &> log_lgode_gat_sst_f0.txt &  
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --use_gat --save_name=lgode_gat_sst_200 --fourier_coeff=200 &> log_lgode_gat_sst_f200.txt &
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --use_gat --save_name=lgode_gat_sst_500 --fourier_coeff=500 &> log_lgode_gat_sst_f500.txt &


# PGODE without GAT 

# CUDA_VISIBLE_DEVICES=2 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst_0 &> log_pgode_sst_f0_.txt &
# CUDA_VISIBLE_DEVICES=3 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst_200 --fourier_coeff=200 &> log_pgode_sst_f200_.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst_500 --fourier_coeff=500 &> log_pgode_sst_f500_.txt &


# # PGODE with GAT
# CUDA_VISIBLE_DEVICES=6 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --use_gat --dataset=sst_pcs.mat --save_name=pgode_gat_sst &> log_pgode_gat_sst_f0_.txt &
# CUDA_VISIBLE_DEVICES=7 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --use_gat --dataset=sst_pcs.mat --save_name=pgode_gat_sst_200 --fourier_coeff=200 &> log_pgode_gat_sst_f200_.txt &
CUDA_VISIBLE_DEVICES=5 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --use_gat --dataset=sst_pcs.mat --save_name=pgode_gat_sst_500 --fourier_coeff=500 &> log_pgode_gat_sst_f500_.txt &