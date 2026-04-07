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
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=5e-5, help="Starting learning rate.")
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--ckpt_path', type=str, default='results/results_sst/lgode_gat_sst_0', help="Path for save checkpoints") 
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
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--save_name", type=str, default="")
parser.add_argument("--input_file", type=str, default="../data/ocean_timeseries.csv")
parser.add_argument("--eval_criterion", type=str, default="all") # if all, this means report rmse for all dimensions together, else we stop by nino3.4 
parser.add_argument("--train_loss", type=str, default="all") # if all, this means use training loss based on all nodes, else just focus on nino3.4
parser.add_argument("--dataset", type=str, default="data/ocean_timeseries.csv")
parser.add_argument("--single_target", action="store_true")
parser.add_argument("--use_gat", action="store_true")
parser.add_argument("--attention_only", type=bool, default=True)

# 11/18/2024
# NOTE: try adding fourier loss and change the periodic embedding
parser.add_argument("--fourier_coeff", type=float, default=0.)
parser.add_argument("--period", type=int, default=12)
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

    now = datetime.datetime.now()
    date = str(now) # str(uuid.uuid4())
     
    log_path = f"results/{date}_{args.save_name}/output.log"
    ############ Command Related
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    if not os.path.exists(f"results/{date}_{args.save_name}"):
        utils.makedirs(f"results/{date}_{args.save_name}")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args)) 
    

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

    ############ Model Select
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)
    ##################################################################
    # Load checkpoint and evaluate the model 
    ckpt_path = os.path.join(args.ckpt_path, "model.ckpt")
    print(f"loading the model from {ckpt_path}")
    utils.get_ckpt_model(ckpt_path, model, device)
    gnn_model = model.encoder_z0  # NOTE: we only need the gnn module inside
    
    print(f"obtained the gnn model: {gnn_model}")

    n_batches = test_batch

    def get_next_batch_new(dataloader,device):
        data_dict = dataloader.__next__()
        #device_now = data_dict.batch.device
        return data_dict.to(device)

    for i in tqdm(range(n_batches)):
        batch_dict_encoder = get_next_batch_new(test_encoder, device)
        
        attention_score = gnn_model.get_attention_score(batch_dict_encoder.x, batch_dict_encoder.edge_attr, 
                                                        batch_dict_encoder.edge_index, batch_dict_encoder.pos, 
                                                        batch_dict_encoder.edge_same)
        print(attention_score)
        print(f"attention_score shape = {attention_score[0].shape}")