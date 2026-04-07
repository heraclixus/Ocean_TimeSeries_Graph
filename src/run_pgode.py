import os
import sys
import time

from lgode.lib.new_dataLoader import ParseData, inverse_normalize
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import pgode.lib.utils as utils
from torch.distributions.normal import Normal
from pgode.lib.create_latent_ode_model import create_LatentODE_model
from pgode.lib.utils import compute_loss_all_batches
from utils_pca import reconstruct_enso
import matplotlib.pyplot as plt
from pygtemporal_models.pyg_temp_dataset import batch_data_to_timeseries
import xarray as xr
import xskillscore as xs
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from utils_visualization_forecast import plot_enso_forecast_vs_real, plot_enso_anomaly_correlation, plot_channel_rmse, plot_enso_anomaly_rmse


def plot_rmse_and_rmse_recon(save_path, rmses_train, rmses_test, rmses_train_reconstructed, rmses_test_reconstructed):
    fig_rmse = plt.figure(figsize=(10,5))
    plt.plot(rmses_train, label="Train RMSE")
    plt.plot(rmses_test, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Train and Test RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"rmses.png"))
    plt.close()
    # visualize train and test rmses reconstructed
    fig_rmse_reconstructed = plt.figure(figsize=(10,5))
    plt.plot(rmses_train_reconstructed, label="Train RMSE Reconstructed")
    plt.plot(rmses_test_reconstructed, label="Test RMSE Reconstructed")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Reconstructed")
    plt.title("Train and Test RMSE Reconstructed")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"rmses_reconstructed.png"))
    plt.close()



# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=20,
                    help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=1e-5, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=96)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1, help="Random_seed")
parser.add_argument('--data', type=str, default='ocean', help="spring,charged,motion")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--latents_global', type=int, default=4, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--extrap', type=str,default="True", help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample-percent-train', type=float, default=1.0,help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=1.0,help='Percentage of testing observtaion data')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-2, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')
parser.add_argument('--condition_num', type=int, default=12, help='condition num ')
parser.add_argument('--extrap_num', type=int, default=12, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")
parser.add_argument('--moe_level', type=str, default="node_level")

parser.add_argument("--use_cosine", action="store_true")
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--use_warmup", action="store_true")


parser.add_argument('--expert_num', type=int, default=5)
parser.add_argument('--use_bn', type=int, default=0)
parser.add_argument('--num_sys_paras', type=int, default=4)
parser.add_argument('--MI_coef', type=float, default=0.01)
parser.add_argument('--save_traj', type=int, default=0)
parser.add_argument('--disen_coef', type=float, default=0.01)
parser.add_argument('--gate_use_global', type=int, default=1)
parser.add_argument('--wo_local', type=int, default=0)

parser.add_argument('--cond_len', type=int, default=6)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--test', type=int, default=0)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--save_name", type=str, default="")
parser.add_argument("--input_file", type=str, default="../data/ocean_timeseries.csv")
parser.add_argument("--eval_criterion", type=str, default="all") # if all, this means report rmse for all dimensions together, else we stop by nino3.4 
parser.add_argument("--train_loss", type=str, default="all") # if all, this means use training loss based on all nodes, else just focus on nino3.4
parser.add_argument("--dataset", type=str, default="data/ocean_timeseries.csv")
parser.add_argument("--single_target", action="store_true")

# 11/18/2024
# NOTE: try adding fourier loss and change the periodic embedding
parser.add_argument("--fourier_coeff", type=float, default=0.)
parser.add_argument("--period", type=int, default=12)

# 12/25/2024
parser.add_argument("--use_gat", action="store_true")
# as from the README, we have 5 levels 
parser.add_argument("--feature_set", type=int, default=1) # 
#parser.add_argument('--alias', type=str, default="run")
parser.add_argument("--n_pcs", type=int, default=20)


args = parser.parse_args()
assert(int(args.rec_dims%args.n_heads) ==0)
args.total_ode_step=args.condition_num+args.extrap_num
args.total_ode_step_train=args.condition_num+args.extrap_num
args.total_ode_step = 73
args.suffix = '' 
criterion = args.eval_criterion


def lr_lambda(epoch):
    if epoch < 5:
        return epoch / 5  # Linear warm-up
    return 1  # Default multiplier after warm-up


############ CPU AND GPU related, Mode related, Dataset Related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:0")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-"*80)
    args.mode = "extrap"
elif args.extrap=="False":
    print("Running interp mode" + "-" * 80)
    args.mode="interp"


save_name = f"window={args.cond_len}_horizon={args.pred_len}_npcs={args.n_pcs}_batch={args.batch_size}"
if args.use_gat:
    save_name += f"_gat"

if args.use_scheduler:
    save_name += "_scheduler"

if args.use_cosine:
    save_name += "_cosine"

if args.use_warmup:
    save_name += "_warmup"

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    ############ Saving Path and Preload.
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)


    ############ Loading Data
    print(args)
    print("Loading dataset: " + args.dataset)
    dataloader = ParseData(mode=args.mode, args=args)
    train_encoder,train_decoder, train_graph, train_batch = dataloader.load_data(sample_percent=args.sample_percent_train,
                                                                                batch_size=args.batch_size,data_type="train")

    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                              batch_size=1,data_type="test")
    
    # dataloader.plot_std()
    # examine the data loader 
    # train_batch = next(train_encoder)
    # print(train_batch)
    # print(train_batch.x.shape)
    # print(train_batch.y.shape)

    # test_batch = next(test_encoder)
    # print(test_batch)
    # print(test_batch.x.shape)
    # print(test_batch.y.shape)
    # exit(0)


    train_original_max = dataloader.original_max
    train_original_min = dataloader.original_min


    # print(f"train_original_max = {train_original_max.shape}, train_original_min = {train_original_min.shape}")

    input_dim = dataloader.feature

    if args.single_target:
        nino_col = dataloader.nino_feature_index
        n_features = len(dataloader.features)

    else:
        nino_col = None
        n_features = None

    ############ Command Related
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    ############ Model Select
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)


    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        #exit()

    ##################################################################
    # Training

    log_path = ("logs/" + args.alias + "_" + args.z0_encoder+ "_" + args.data + "_cond" +str(args.condition_num)
                + "_expert" +str(args.expert_num)
                + "_MIcoef" +str(args.MI_coef) + "_" + args.mode + "_" + str(experimentID) + ".log")
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
    if args.use_warmup:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.patience, T_mult=2, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    wait_until_kl_inc = 10
    best_test_rmse = np.inf
    best_test_rmse_val = np.inf
    best_val_mse = np.inf
    n_iters_to_viz = 1


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):

        optimizer.zero_grad()
        # understand the data
        

        train_res, pred_y = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                             n_traj_samples=3, kl_coef=kl_coef)

        loss = train_res["loss"]
        loss.backward()

        # filter NaN grad
        for param in optimizer.param_groups:
            for p in param['params']:
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        logger.info('filtering NaN grad!')
                        p.grad[torch.isnan(p.grad)] = 0

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["mse"],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"], \
            train_res["disen_loss"], pred_y

    def train_epoch(epo):
        model.train()
        loss_list = []
        loss_disen_list = []
        mse_list = []

        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()
        total_true_y = []
        total_pred_y = []

        total_true_enso = []
        total_pred_enso = [] 

        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 10

            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)


            loss, mse,likelihood,kl_first_p,std_first_p,\
                disen_loss, pred_y = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,
                                                         batch_dict_graph,kl_coef)

        
            #saving results
            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(
               likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)
            loss_disen_list.append(disen_loss)

            pred_y = pred_y.detach().cpu().numpy()
            true_y = batch_dict_decoder['data'].detach().cpu().numpy()
            # total_pred_y.append(pred_y)
            # total_true_y.append(true_y)

            # print(f"pred_y = {pred_y.shape}, true_y = {true_y.shape}")
            # convert to time series format and reconstruct
            pred_y = pred_y.reshape(-1, 1, args.n_pcs, args.pred_len)
            true_y = true_y.reshape(-1, 1, args.n_pcs, args.pred_len)
            pred_y = batch_data_to_timeseries(pred_y)
            true_y = batch_data_to_timeseries(true_y)

            

            pred_enso, true_enso = reconstruct_enso(pcs=inverse_normalize(pred_y, train_original_max, train_original_min), 
                                                    real_pcs=inverse_normalize(true_y,train_original_min, train_original_min),
                                                    top_n_pcs=args.n_pcs, flag="train")
            
            total_true_enso.append(true_enso)
            total_pred_enso.append(pred_enso)
            total_true_y.append(true_y)
            total_pred_y.append(pred_y)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()
        if args.use_scheduler:
            scheduler.step()
        train_true_y, train_pred_y = np.concatenate(total_true_y, axis=0), np.concatenate(total_pred_y, axis=0)
        train_true_enso, train_pred_enso = np.concatenate(total_true_enso, axis=0), np.concatenate(total_pred_enso, axis=0)
        true = inverse_normalize(train_true_y, train_original_max, train_original_min)
        pred = inverse_normalize(train_pred_y, train_original_max, train_original_min)

       #  print(f"true = {true.shape}, pred = {pred.shape}")
        # print(f"train_true_enso = {train_true_enso.shape}, train_pred_enso = {train_pred_enso.shape}")
        
        if nino_col is not None:
            true_reshaped = true.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
            pred_reshaped = pred.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
            rmse_nino = np.sqrt(np.mean(np.square(true_reshaped-pred_reshaped)))
            mape = np.mean(np.abs(true_reshaped-pred_reshaped)/true_reshaped)


        mape = np.mean(np.abs(true - pred)/true)
        rmse = np.sqrt(np.mean((true - pred)**2))
        rmse_enso = np.sqrt(np.mean((train_true_enso - train_pred_enso)**2))
        # print(f"rmse_enso = {rmse_enso}")

        if nino_col is not None:
            message_train = ('Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | Loss_disen {:.4f} | '
                         'MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}| | RMSE_NINO {:.6F} | MAPE {:.6f} | ').format(
            epo,
            np.mean(loss_list), np.mean(loss_disen_list),  np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list), rmse_nino, mape)
        else:   
            message_train = ('Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | Loss_disen {:.4f} | '
                         'MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}| | RMSE {:.6F} | MAPE {:.6f} | RMSE_ENSO {:.4f}').format(
            epo,
            np.mean(loss_list), np.mean(loss_disen_list),  np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list), rmse, mape, rmse_enso)
        if nino_col is not None:
            rmse = rmse_nino


        return message_train, kl_coef, true, pred, [rmse, rmse_enso]
    
    import datetime
    now = datetime.datetime.now()
    date = str(now) # str(uuid.uuid4())

    os.makedirs(f"results/pgode/{save_name}", exist_ok=True)
    np.save(f"results/pgode/{save_name}/train_original_max.npy", train_original_max)
    np.save(f"results/pgode/{save_name}/train_original_min.npy", train_original_min)

    cumulative_patience = 0
    best_epo = 0
    rmses_train, rmses_test = [],[]
    reconstructed_rmses_train, reconstructed_rmses_test = [],[]
    bestmodel = None
    kl_coef = 0.0 

    for epo in range(1, args.niters + 1):
        time_start = time.time()
        
        message_train, kl_coef, train_true_y, train_pred_y, train_metrics = train_epoch(epo)

        train_rmse, train_rmse_recon = train_metrics
        rmses_train.append(train_rmse)
        reconstructed_rmses_train.append(train_rmse_recon)
        
        # message_train = "deug"
        # kl_coef = 0
        
        logger.info("cost_time: {:.6f} s".format(time.time() - time_start))

        if epo % n_iters_to_viz == 0:
            model.eval()
            test_res, test_true_y, test_pred_y = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)
            
            test_true_y = np.squeeze(test_true_y)
            test_pred_y = np.squeeze(test_pred_y)            
            # compute reconstruction
            # (batch, 20, 24) 
            test_true_y = test_true_y.reshape(-1, args.n_pcs, args.pred_len)
            test_pred_y = test_pred_y.reshape(-1, args.n_pcs, args.pred_len)

            test_true = batch_data_to_timeseries(test_true_y)
            test_pred = batch_data_to_timeseries(test_pred_y)

            test_true = inverse_normalize(test_true, train_original_max, train_original_min)
            test_pred = inverse_normalize(test_pred, train_original_max, train_original_min)


            test_enso, test_enso_pred = reconstruct_enso(pcs=test_pred, real_pcs=test_true, top_n_pcs=args.n_pcs, flag="test")
                        
            rmse = np.sqrt(np.mean((test_true - test_pred)**2))
            mape = np.mean(np.abs(test_true - test_pred)/test_true)
            rmse_recon = np.sqrt(np.mean((test_enso - test_enso_pred)**2))

            rmses_test.append(rmse)
            reconstructed_rmses_test.append(rmse_recon)

            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}| | RMSE {:.6F} | MAPE {:.6f} | RMSE_RECON {:.3f} '.format(
            epo,
            test_res["loss"], test_res["mse"], test_res["likelihood"],
            test_res["kl_first_p"], test_res["std_first_p"], rmse, mape, rmse_recon)

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            print("data: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))
            

            if rmse < best_test_rmse:
                best_epo = epo
                best_test_rmse = rmse
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best rmse {:.6f}|'.format(epo,
                                                                                                        best_test_rmse)
                logger.info(message_best)
                os.makedirs(f"results/pgode/{save_name}", exist_ok=True)
                ckpt_path = os.path.join(f"results/pgode/{save_name}/model.ckpt")
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)
                
                cumulative_patience = 0 
                np.save(f"results/pgode/{save_name}/test_pred.npy", test_pred)
                np.save(f"results/pgode/{save_name}/test_true.npy", test_true)
                np.save(f"results/pgode/{save_name}/test_pred_y.npy", test_pred_y)
                np.save(f"results/pgode/{save_name}/test_true_y.npy", test_true_y)
                np.save(f"results/pgode/{save_name}/test_pred_enso.npy", test_enso_pred)
                np.save(f"results/pgode/{save_name}/test_enso.npy", test_enso)

                # plot forecast skills

                test_enso_pred_skill, test_enso_true_skill = [],[] 
                for i in range(len(test_pred_y)):
                    test_pred_y_i = test_pred_y[i].T # (24, 20)
                    test_true_y_i = test_true_y[i].T # (24, 20)
                    # (24)
                    test_true_enso_i, test_pred_enso_i = reconstruct_enso(pcs=inverse_normalize(test_pred_y_i, train_original_max, train_original_min), 
                                                                        real_pcs=inverse_normalize(test_true_y_i,train_original_min, train_original_min),
                                                                        top_n_pcs=args.n_pcs, flag="test")
                    test_enso_pred_skill.append(test_pred_enso_i)
                    test_enso_true_skill.append(test_true_enso_i)
                # (b, 24)
                test_enso_pred_skill = np.stack(test_enso_pred_skill)
                test_enso_true_skill = np.stack(test_enso_true_skill)

                plot_enso_anomaly_correlation(prediction=test_enso_pred_skill, 
                                               test=test_enso_true_skill, 
                                               model_name="PGODE", 
                                               save_path=f"results/pgode/{save_name}")

                # TODO: hack                
                plot_enso_forecast_vs_real(prediction=test_enso_true_skill, 
                                           test=test_enso_pred_skill,
                                           model_name="PGODE",
                                           save_path=f"results/pgode/{save_name}")
                
                plot_enso_anomaly_rmse(prediction=test_enso_true_skill, 
                                       test=test_enso_pred_skill,
                                       model_name="PGODE",
                                       save_path=f"results/pgode/{save_name}")
                
                test_pred_y = test_pred_y.transpose(0,2,1)
                test_true_y = test_true_y.transpose(0,2,1)

                plot_channel_rmse(prediction=test_pred_y, test=test_true_y, model_name="PGODE", n_pcs=args.n_pcs, save_path=f"results/pgode/{save_name}")

                np.save(f"results/pgode/{save_name}/test_enso_pred_skill", test_enso_pred_skill)
                np.save(f"results/pgode/{save_name}/test_enso_true_skill", test_enso_true_skill)
                
            else:
                cumulative_patience += 1 

            if cumulative_patience >= args.patience:
                print(f"Early stopping: curernt best rmse is {best_test_rmse} at epoch {best_epo}.")
                torch.cuda.empty_cache()
                break
                
            
            torch.cuda.empty_cache()
        

    plot_rmse_and_rmse_recon(save_path=f"results/pgode/{save_name}", 
                             rmses_train=rmses_train, 
                             rmses_test=rmses_test, 
                             rmses_train_reconstructed=reconstructed_rmses_train, 
                             rmses_test_reconstructed=reconstructed_rmses_test)

