# configs 
# for node, we have --add_sin_cos and --ode_encoder_decoder
# for nsde we have additional paramter
CUDA_VISIBLE_DEVICES=0 nohup python run_baseline_models.py --use_normalization --model_name=graphode --coarse_grain_factor=5 &> logs_report_grid/log_graphode_64_6_coarse_5.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_baseline_models.py --use_normalization --model_name=graphode --coarse_grain_factor=4 &> logs_report_grid/log_graphode_64_6_coarse_4.txt & 
CUDA_VISIBLE_DEVICES=0 nohup python run_baseline_models.py --use_normalization --model_name=graphode --coarse_grain_factor=3 &> logs_report_grid/log_graphode_64_6_coarse_3.txt & 
CUDA_VISIBLE_DEVICES=0 nohup python run_baseline_models.py --use_normalization --model_name=graphode --coarse_grain_factor=2 &> logs_report_grid/log_graphode_64_6_coarse_2.txt & 
CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --use_normalization --model_name=graphode --coarse_grain_factor=1 &> logs_report_grid/log_graphode_64_6_coarse_1.txt & 

# for model in {graphode,}; do
#     for window in {6,}; do
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}-periodic_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}-periodic_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}-periodic_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}-periodic_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}-periodic_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}-periodic_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}_weighted_96_${window}_sin-cos.txt &

        # CUDA_VISIBLE_DEVICES=2 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=2 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --n_pcs=5 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic_weighted_96_${window}_sin-cos.txt &

        # CUDA_VISIBLE_DEVICES=4 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=4 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=4 nohup python run_baseline_models.py --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}_weighted_96_${window}_sin-cos.txt & 
        
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --n_pcs=20 --use_periodic_activation --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic_weighted_96_${window}_sin-cos.txt &               
#     done
# done 



# for model in {graphode,}; do
#     for window in {6,}; do
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}-encoder-decoder_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}-encoder-decoder_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}-encoder-decoder_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report_5/log_${model}-encoder-decoder-periodic_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report_5/log_${model}-encoder-decoder-periodic_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report_5/log_${model}-encoder-decoder-periodic_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}-encoder-decoder_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}-encoder-decoder_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}-encoder-decoder_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window &> logs_report/log_${model}-encoder-decoder-periodic_weighted_32_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window &> logs_report/log_${model}-encoder-decoder-periodic_weighted_64_${window}.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window &> logs_report/log_${model}-encoder-decoder-periodic_weighted_96_${window}.txt &

        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}-encoder-decoder_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}-encoder-decoder_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}-encoder-decoder_weighted_96_${window}_sin-cos.txt &

        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic-encoder-decoder_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic-encoder-decoder_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=7 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=5 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report_5/log_${model}-periodic-encoder-decoder_weighted_96_${window}_sin-cos.txt &

        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}-encoder-decoder_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=1 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}-encoder-decoder_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=2 nohup python run_baseline_models.py --ode_encoder_decoder --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}-encoder-decoder_weighted_96_${window}_sin-cos.txt & 

        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=32 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic-encoder-decoder_weighted_32_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=4 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=64 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic-encoder-decoder_weighted_64_${window}_sin-cos.txt &
        # CUDA_VISIBLE_DEVICES=3 nohup python run_baseline_models.py --ode_encoder_decoder --use_periodic_activation --n_pcs=20 --use_loss_weights --use_normalization --model_name=$model --batch_size=96 --window=$window --add_sin_cos &> logs_report/log_${model}-periodic-encoder-decoder_weighted_96_${window}_sin-cos.txt &               
#     done
# done 

#use_periodic_activation