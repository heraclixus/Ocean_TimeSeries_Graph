CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=stemgnn --use_normalization &> log_stemgnn.txt &
CUDA_VISIBLE_DEVICSE=1 nohup python run_pygtemporal_models.py --model_name=agcrn --use_normalization &> log_agcrnn.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=fgnn --use_normalization &> log_fgnn.txt &
CUDA_VISIBLE_DEVICSE=1 nohup python run_pygtemporal_models.py --model_name=mtgnn --use_normalization &> log_mtgnn.txt & 

CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=stemgnn &> log_stemgnn_nonorm.txt &
CUDA_VISIBLE_DEVICSE=1 nohup python run_pygtemporal_models.py --model_name=agcrn &> log_agcrnn_nonorm.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_pygtemporal_models.py --model_name=fgnn &> log_fgnn_nonorm.txt &
CUDA_VISIBLE_DEVICSE=1 nohup python run_pygtemporal_models.py --model_name=mtgnn &> log_mtgnn_nonorm.txt & 