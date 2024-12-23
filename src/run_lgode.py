import os
import sys
from lgode.lib.new_dataLoader import ParseData, inverse_normalize
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.optim as optim
import lgode.lib.utils as utils
from torch.distributions.normal import Normal
from lgode.lib.create_latent_ode_model import create_LatentODE_model
from lgode.lib.utils import compute_loss_all_batches
import matplotlib.pyplot as plt 
import random
import datetime
# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=100000)
parser.add_argument('--lr',  type=float, default=5e-5, help="Starting learning rate.")
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints") 
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--data', type=str, default='ocean', help="spring,charged,motion")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=32, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--extrap', type=str,default="True", help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample-percent-train', type=float, default=1,help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=1,help='Percentage of testing observtaion data')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--cond_len', type=int, default=12)
parser.add_argument('--pred_len', type=int, default=24)
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


# as from the README, we have 5 levels 
parser.add_argument("--feature_set", type=int, default=1) # 
#parser.add_argument('--alias', type=str, default="run")


args = parser.parse_args()
assert(int(args.rec_dims%args.n_heads) ==0)

args.total_ode_step = 73
args.suffix = '' 
criterion = args.eval_criterion



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




#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    ############ Saving Path and Preload.
    file_name = os.path.basename(__file__)[:-3]  # run_models
    

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = args.random_seed# int(SystemRandom().random() * 100000)


    ############ Loading Data
    print("Loading dataset: " + args.dataset)
    dataloader = ParseData(mode=args.mode, args =args)
    test_encoder, test_decoder, test_graph, test_batch, test_original_max, test_original_min = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                              batch_size=args.batch_size,
                                                                              data_type="test")
    train_encoder,train_decoder, train_graph,train_batch, train_original_max, train_original_min = dataloader.load_data(sample_percent=args.sample_percent_train,batch_size=args.batch_size,data_type="train")
     
    train_original_max = train_original_max.reshape(-1,1,1)
    train_original_min = train_original_min.reshape(-1,1,1)
    test_original_max = test_original_max.reshape(-1,1,1)
    test_original_min = test_original_min.reshape(-1,1,1)


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
    now = datetime.datetime.now()
    date = str(now) # str(uuid.uuid4())
     
    log_path = f"results/{date}_{args.save_name}/output.log"

    if not os.path.exists(f"results/{date}_{args.save_name}"):
        utils.makedirs(f"results/{date}_{args.save_name}")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args)) 

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_rmse = np.inf
    n_iters_to_viz = 1


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):

        optimizer.zero_grad()
        train_res, pred_y = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                             n_traj_samples=3, kl_coef=kl_coef)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["mse"],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"], pred_y

    def train_epoch(epo):
        model.train()
        loss_list = []
        rmse_list = [] 
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

        total_true_y = []
        total_pred_y = []
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

            loss, mse, likelihood, kl_first_p, std_first_p, pred_y = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef)

            pred_y = pred_y.detach().cpu().numpy()
            true_y = batch_dict_decoder['data'].detach().cpu().numpy()
            total_pred_y.append(pred_y)
            total_true_y.append(true_y)
            #saving results
            loss_list.append(loss), rmse_list.append(np.sqrt(mse)), likelihood_list.append(
               likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step() 

        train_true_y, train_pred_y = np.concatenate(total_true_y, axis=0), np.concatenate(total_pred_y, axis=0)

        # n_seq x N x T
        true = inverse_normalize(train_true_y, train_original_max, train_original_min)
        pred = inverse_normalize(train_pred_y, train_original_max, train_original_min)
        #true = train_true_y 
        #pred = train_pred_y 

        if nino_col is not None:
            true_reshaped = true.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
            pred_reshaped = pred.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
            rmse_nino = np.sqrt(np.mean(np.square(true_reshaped-pred_reshaped)))
            mape = np.mean(np.abs(true_reshaped-pred_reshaped)/true_reshaped)

        mape = np.mean(np.abs(true - pred)/true)
        rmse = np.sqrt( np.mean( np.square(true - pred), axis=2 ) ).mean(1).mean()
        if nino_col is not None:
            message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | RMSE_NINO {:.6F}  | RMSE {:.6F} | MAPE: {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                np.mean(loss_list), rmse_nino, rmse, mape, np.mean(likelihood_list),
                np.mean(kl_first_p_list), np.mean(std_first_p_list))
        else:
            message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | RMSE {:.6F} | MAPE: {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                np.mean(loss_list), rmse, mape, np.mean(likelihood_list),
                    np.mean(kl_first_p_list), np.mean(std_first_p_list))
        
        
        if nino_col is not None:
            rmse = rmse_nino

        return message_train, kl_coef, true, pred, [rmse]
    
    np.save(f"results/{date}_{args.save_name}/test_original_max.npy", test_original_max)
    np.save(f"results/{date}_{args.save_name}/test_original_min.npy", test_original_min)
    np.save(f"results/{date}_{args.save_name}/train_original_max.npy", train_original_max)
    np.save(f"results/{date}_{args.save_name}/train_original_min.npy", train_original_min)



    # NOTE: 11/04 early stopping for training.
    # NOTE: 11/05 add reporting for only EL NINO indices.
    cumulative_patience = 0
    best_epo = 0

    for epo in range(1, args.niters + 1):

        message_train, kl_coef, train_true_y, train_pred_y, train_metrics = train_epoch(epo)
          
        if epo % n_iters_to_viz == 0:
            model.eval()
            test_res, test_true_y, test_pred_y = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)
            
            test_true = inverse_normalize(test_true_y, test_original_max, test_original_min)
            test_pred = inverse_normalize(test_pred_y, test_original_max, test_original_min)

            if nino_col is not None:
                test_true_reshaped = test_true.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
                test_pred_reshaped = test_pred.reshape(-1, n_features, args.pred_len)[:,nino_col,:].flatten()
                rmse_nino = np.sqrt(np.mean(np.square(test_true_reshaped-test_pred_reshaped)))
                mape = np.mean(np.abs(test_true_reshaped-test_pred_reshaped)/test_true_reshaped)
           
            rmse = np.sqrt( np.mean( np.square(test_true - test_pred), axis=2 ) ).mean(1).mean()
            mape = np.mean(np.abs(test_true - test_pred)/test_true)
            if nino_col is not None:
                message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | RMSE_NINO {:.6F}  | RMSE {:.6F} | MAPE: {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    epo,
                    test_res["loss"], rmse_nino, rmse, mape, test_res["likelihood"],
                    test_res["kl_first_p"], test_res["std_first_p"])
            else:
                message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | RMSE {:.6F} | MAPE: {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    epo,
                    test_res["loss"], rmse, mape, test_res["likelihood"],
                    test_res["kl_first_p"], test_res["std_first_p"])

            #logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            print("data: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))

            if criterion == "nino":
                rmse = rmse_nino
            if rmse < best_test_rmse:
                best_epo = epo
                best_test_rmse = rmse
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}|'.format(epo,
                                                                                                        best_test_rmse)
                logger.info(message_best)
                ckpt_path = os.path.join(f"results/{date}_{args.save_name}/model.ckpt")
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)

                # reset patience
                cumulative_patience = 0 

                np.save(f"results/{date}_{args.save_name}/test_pred.npy", test_pred)
                np.save(f"results/{date}_{args.save_name}/test_true.npy", test_true)
                np.save(f"results/{date}_{args.save_name}/train_pred.npy", train_pred_y)
                np.save(f"results/{date}_{args.save_name}/train_true.npy", train_true_y)

            cumulative_patience += 1 
            if cumulative_patience >= args.patience:
                print(f"Early stopping: curernt best rmse is {best_test_rmse} at epoch {best_epo}.")
                torch.cuda.empty_cache()
                break
            torch.cuda.empty_cache()