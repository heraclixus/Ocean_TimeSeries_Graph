import os
import sys
import time

from pgode.lib.new_dataLoader import ParseData
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

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr',  type=float, default=5e-4, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1, help="Random_seed")
parser.add_argument('--data', type=str, default='spring', help="spring,charged,motion")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--latents_global', type=int, default=4, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--extrap', type=str,default="False", help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample-percent-train', type=float, default=1.0,help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=1.0,help='Percentage of testing observtaion data')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')
parser.add_argument('--condition_num', type=int, default=12, help='condition num ')
parser.add_argument('--extrap_num', type=int, default=12, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")
parser.add_argument('--moe_level', type=str, default="node_level")

parser.add_argument('--expert_num', type=int, default=5)
parser.add_argument('--use_bn', type=int, default=0)
parser.add_argument('--num_sys_paras', type=int, default=4)
parser.add_argument('--MI_coef', type=float, default=0.01)
parser.add_argument('--save_traj', type=int, default=0)
parser.add_argument('--disen_coef', type=float, default=0.01)
parser.add_argument('--gate_use_global', type=int, default=1)
parser.add_argument('--wo_local', type=int, default=0)


args = parser.parse_args()
assert(int(args.rec_dims%args.n_heads) ==0)



if args.data == "spring_1000_10_24":
    args.dataset = 'data/springs_1000_10_24'
    args.suffix = '_springs10'
    args.total_ode_step=24
elif args.data == "spring_1000_10_12":
    args.dataset = 'data/springs_1000_10_12'
    args.suffix = '_springs10'
    args.total_ode_step=12
elif args.data == "spring_1000_10_48":
    args.dataset = 'data/springs_1000_10_48'
    args.suffix = '_springs10'
    args.total_ode_step=48
elif args.data == "charged_1000_10_24":
    args.dataset = 'data/charged_1000_10_24'
    args.suffix = '_charged10'
    args.total_ode_step=24
elif args.data == "charged_1000_10_12":
    args.dataset = 'data/charged_1000_10_12'
    args.suffix = '_charged10'
    args.total_ode_step=12
elif args.data == "charged_1000_10_48":
    args.dataset = 'data/charged_1000_10_48'
    args.suffix = '_charged10'
    args.total_ode_step=48

args.total_ode_step=args.condition_num+args.extrap_num
args.total_ode_step_train=args.condition_num+args.extrap_num

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
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)


    ############ Loading Data
    print("Loading dataset: " + args.dataset)
    dataloader = ParseData(args.dataset,suffix=args.suffix,mode=args.mode, args =args)
    test_encoder, test_decoder, test_graph, test_para, test_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                              batch_size=args.batch_size,
                                                                              data_type="test")
    train_encoder,train_decoder, train_graph, train_para, train_batch = dataloader.load_data(sample_percent=args.sample_percent_train,batch_size=args.batch_size,data_type="train")
    val_encoder, val_decoder, val_graph, val_para, val_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                          batch_size=args.batch_size,
                                                                          data_type="val")


    input_dim = dataloader.feature

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_mse = np.inf
    best_test_mse_val = np.inf
    best_val_mse = np.inf
    n_iters_to_viz = 1


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,batch_dict_para,kl_coef):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,batch_dict_para,
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
            train_res["disen_loss"], train_res["sys_loss"]

    def train_epoch(epo):
        model.train()
        loss_list = []
        loss_disen_list = []
        loss_sys_list = []
        mse_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

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

            batch_dict_para = utils.get_next_batch_new(train_para, device)

            loss, mse,likelihood,kl_first_p,std_first_p,\
                disen_loss,sys_loss = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,
                                                         batch_dict_graph,batch_dict_para,kl_coef)

        
            #saving results
            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(
               likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)
            loss_disen_list.append(disen_loss)
            loss_sys_list.append(sys_loss)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()
        

        message_train = ('Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | Loss_disen {:.4f} | '
                         'Loss_sys {:.4f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|').format(
            epo,
            np.mean(loss_list), np.mean(loss_disen_list), np.mean(loss_sys_list),  np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list))


        return message_train, kl_coef


    for epo in range(1, args.niters + 1):
        time_start = time.time()
        message_train, kl_coef = train_epoch(epo)
        logger.info("cost_time: {:.6f} s".format(time.time() - time_start))

        if epo % n_iters_to_viz == 0:
            model.eval()
            val_res = compute_loss_all_batches(model, val_encoder, val_graph, val_decoder, val_para,
                                                n_batches=val_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)
            message_val = 'Epoch {:04d} [Val seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                val_res["loss"], val_res["mse"], val_res["likelihood"],
                val_res["kl_first_p"], val_res["std_first_p"])

            test_res = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder, test_para,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)
            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                test_res["loss"], test_res["mse"], test_res["likelihood"],
                test_res["kl_first_p"], test_res["std_first_p"])

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_val)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            print("data: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))

            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_test_mse_val = test_res["mse"]
                message_best_test_val = 'Best val | Epoch {:04d} [Test seq (cond on sampled tp)] | mse {:.6f}|'.format(epo,
                                                                                                        best_test_mse_val)
                logger.info(message_best_test_val)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.z0_encoder + "_" + args.data + "_expert" +str(args.expert_num) + "_cond" + str(
                    args.condition_num) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse_val) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)

            if test_res["mse"] < best_test_mse:
                best_test_mse = test_res["mse"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}|'.format(epo,
                                                                                                        best_test_mse)
                logger.info(message_best)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.z0_encoder + "_" + args.data + "_expert" +str(args.expert_num) + "_cond" + str(
                    args.condition_num) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)

            torch.cuda.empty_cache()

    logger.info(message_best_test_val)
    logger.info(message_best)


