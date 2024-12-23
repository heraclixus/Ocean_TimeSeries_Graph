# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=sst_0 --fourier_coeff=0 &> log_sst_f0.txt &  
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=sst_1 --fourier_coeff=1 &> log_sst_f1.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=sst_2 --fourier_coeff=2 &> log_sst_f2.txt &
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=sst_3 --fourier_coeff=3 &> log_sst_f3.txt &
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=sst_4 --fourier_coeff=4 &> log_sst_f4.txt &


# CUDA_VISIBLE_DEVICES=3 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst &> log_pgode_sst.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst --fourier_coeff=100 &> log_pgode_sst_f100.txt &
CUDA_VISIBLE_DEVICES=4 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst --fourier_coeff=200 &> log_pgode_sst_f200.txt &
CUDA_VISIBLE_DEVICES=6 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst --fourier_coeff=500 &> log_pgode_sst_f500.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_pgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=pgode_sst --fourier_coeff=1000 &> log_pgode_sst_f1000.txt &
