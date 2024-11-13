CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=1 --save_name=cat1_nino --eval_criterion=nino &> log_cat1_nino.txt &
CUDA_VISIBLE_DEVICES=1 nohup python run_lgode.py --feature_set=2 --save_name=cat2_nino --eval_criterion=nino &> log_cat2_nino.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_lgode.py --feature_set=3 --save_name=cat3_nino --eval_criterion=nino &> log_cat3_nino.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_lgode.py --feature_set=4 --save_name=cat4_nino --eval_criterion=nino &> log_cat4_nino.txt &
CUDA_VISIBLE_DEVICES=4 nohup python run_lgode.py --feature_set=5 --save_name=cat5_nino --eval_criterion=nino &> log_cat5_nino.txt &

CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=1 --save_name=cat1 &> log_cat1.txt &
CUDA_VISIBLE_DEVICES=6 nohup python run_lgode.py --feature_set=2 --save_name=cat2 &> log_cat2.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_lgode.py --feature_set=3 --save_name=cat3 &> log_cat3.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_lgode.py --feature_set=4 --save_name=cat4 &> log_cat4.txt &
CUDA_VISIBLE_DEVICES=5 nohup python run_lgode.py --feature_set=5 --save_name=cat5 &> log_cat5.txt &
