CUDA_VISIBLE_DEVICES=0 nohup python test_lgode.py --input_file=../data/sst_pcs.mat --dataset=sst_pcs.mat --save_name=lgode_sst_0_test --fourier_coeff=0 &> log_lgode_debug.txt &
