# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino --eval_criterion=nino &> log_cat1_nino.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino --eval_criterion=nino &> log_cat2_nino.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino --eval_criterion=nino &> log_cat3_nino.txt &
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino --eval_criterion=nino &> log_cat4_nino.txt &
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino --eval_criterion=nino &> log_cat5_nino.txt &

# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=1 --save_name=cat1 &> log_cat1.txt &
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=2 --save_name=cat2 &> log_cat2.txt &
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=3 --save_name=cat3 &> log_cat3.txt &
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=4 --save_name=cat4 &> log_cat4.txt &
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5 &> log_cat5.txt &


# experiments to run different configuration of fourier coeffs and periods
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_112 --eval_criterion=nino --fourier_coeff=1 --period=12 &> log_cat1_nino_112.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_124 --eval_criterion=nino --fourier_coeff=1 --period=24 &> log_cat1_nino_124.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_110000 --eval_criterion=nino --fourier_coeff=1 --period=10000 &> log_cat1_nino_110000.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_212 --eval_criterion=nino --fourier_coeff=2 --period=12 &> log_cat1_nino_212.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_224 --eval_criterion=nino --fourier_coeff=2 --period=24 &> log_cat1_nino_224.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_210000 --eval_criterion=nino --fourier_coeff=2 --period=10000 &> log_cat1_nino_210000.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_312 --eval_criterion=nino --fourier_coeff=3 --period=12 &> log_cat1_nino_312.txt & 
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_324 --eval_criterion=nino --fourier_coeff=3 --period=24 &> log_cat1_nino_324.txt & 
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_310000 --eval_criterion=nino --fourier_coeff=3 --period=10000 &> log_cat1_nino_310000.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_012 --eval_criterion=nino --fourier_coeff=0 --period=12 &> log_cat1_nino_012.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_024 --eval_criterion=nino --fourier_coeff=0 --period=24 &> log_cat1_nino_024.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino_010000 --eval_criterion=nino --fourier_coeff=0 --period=10000 &> log_cat1_nino_010000.txt &


# experiments to run different configuration of fourier coeffs and periods
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_112 --eval_criterion=nino --fourier_coeff=1 --period=12 &> log_cat2_nino_112.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_124 --eval_criterion=nino --fourier_coeff=1 --period=24 &> log_cat2_nino_124.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_110000 --eval_criterion=nino --fourier_coeff=1 --period=10000 &> log_cat2_nino_110000.txt & 
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_212 --eval_criterion=nino --fourier_coeff=2 --period=12 &> log_cat2_nino_212.txt & 
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_224 --eval_criterion=nino --fourier_coeff=2 --period=24 &> log_cat2_nino_224.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_210000 --eval_criterion=nino --fourier_coeff=2 --period=10000 &> log_cat2_nino_210000.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_312 --eval_criterion=nino --fourier_coeff=3 --period=12 &> log_cat2_nino_312.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_324 --eval_criterion=nino --fourier_coeff=3 --period=24 &> log_cat2_nino_324.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_310000 --eval_criterion=nino --fourier_coeff=3 --period=10000 &> log_cat2_nino_310000.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_012 --eval_criterion=nino --fourier_coeff=0 --period=12 &> log_cat2_nino_012.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_024 --eval_criterion=nino --fourier_coeff=0 --period=24 &> log_cat2_nino_024.txt & 
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino_010000 --eval_criterion=nino --fourier_coeff=0 --period=10000 &> log_cat2_nino_010000.txt &


# experiments to run different configuration of fourier coeffs and periods
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_112 --eval_criterion=nino --fourier_coeff=1 --period=12 &> log_cat3_nino_112.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_124 --eval_criterion=nino --fourier_coeff=1 --period=24 &> log_cat3_nino_124.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_110000 --eval_criterion=nino --fourier_coeff=1 --period=10000 &> log_cat3_nino_110000.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_212 --eval_criterion=nino --fourier_coeff=2 --period=12 &> log_cat3_nino_212.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_224 --eval_criterion=nino --fourier_coeff=2 --period=24 &> log_cat3_nino_224.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_210000 --eval_criterion=nino --fourier_coeff=2 --period=10000 &> log_cat3_nino_210000.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_312 --eval_criterion=nino --fourier_coeff=3 --period=12 &> log_cat3_nino_312.txt & 
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_324 --eval_criterion=nino --fourier_coeff=3 --period=24 &> log_cat3_nino_324.txt & 
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_310000 --eval_criterion=nino --fourier_coeff=3 --period=10000 &> log_cat3_nino_310000.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_012 --eval_criterion=nino --fourier_coeff=0 --period=12 &> log_cat3_nino_012.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_024 --eval_criterion=nino --fourier_coeff=0 --period=24 &> log_cat3_nino_024.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino_010000 --eval_criterion=nino --fourier_coeff=0 --period=10000 &> log_cat3_nino_010000.txt &


# experiments to run different configuration of fourier coeffs and periods
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_112 --eval_criterion=nino --fourier_coeff=1 --period=12 &> log_cat4_nino_112.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_124 --eval_criterion=nino --fourier_coeff=1 --period=24 &> log_cat4_nino_124.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_110000 --eval_criterion=nino --fourier_coeff=1 --period=10000 &> log_cat4_nino_110000.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_212 --eval_criterion=nino --fourier_coeff=2 --period=12 &> log_cat4_nino_212.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_224 --eval_criterion=nino --fourier_coeff=2 --period=24 &> log_cat4_nino_224.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_210000 --eval_criterion=nino --fourier_coeff=2 --period=10000 &> log_cat4_nino_210000.txt & 
# CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_312 --eval_criterion=nino --fourier_coeff=3 --period=12 &> log_cat4_nino_312.txt & 
# CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_324 --eval_criterion=nino --fourier_coeff=3 --period=24 &> log_cat4_nino_324.txt & 
# CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_310000 --eval_criterion=nino --fourier_coeff=3 --period=10000 &> log_cat4_nino_310000.txt & 
# CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_012 --eval_criterion=nino --fourier_coeff=0 --period=12 &> log_cat4_nino_012.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_024 --eval_criterion=nino --fourier_coeff=0 --period=24 &> log_cat4_nino_024.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino_010000 --eval_criterion=nino --fourier_coeff=0 --period=10000 &> log_cat4_nino_010000.txt &


# experiments to run different configuration of fourier coeffs and periods
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_112 --eval_criterion=nino --fourier_coeff=1 --period=12 &> log_cat5_nino_112.txt & 
# CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_124 --eval_criterion=nino --fourier_coeff=1 --period=24 &> log_cat5_nino_124.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_110000 --eval_criterion=nino --fourier_coeff=1 --period=10000 &> log_cat5_nino_110000.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_212 --eval_criterion=nino --fourier_coeff=2 --period=12 &> log_cat5_nino_212.txt & 
# CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_224 --eval_criterion=nino --fourier_coeff=2 --period=24 &> log_cat5_nino_224.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_210000 --eval_criterion=nino --fourier_coeff=2 --period=10000 &> log_cat5_nino_210000.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_312 --eval_criterion=nino --fourier_coeff=3 --period=12 &> log_cat5_nino_312.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_112 --eval_criterion=nino --fourier_coeff=3 --period=24 &> log_cat5_nino_324.txt & 
# CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_112 --eval_criterion=nino --fourier_coeff=3 --period=10000 &> log_cat5_nino_310000.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_012 --eval_criterion=nino --fourier_coeff=0 --period=12 &> log_cat5_nino_012.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_024 --eval_criterion=nino --fourier_coeff=0 --period=24 &> log_cat5_nino_024.txt & 
# CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino_010000 --eval_criterion=nino --fourier_coeff=0 --period=10000 &> log_cat5_nino_010000.txt &

