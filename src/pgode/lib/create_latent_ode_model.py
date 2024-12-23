from pgode.lib.gnn_models import GNN, GateGNN
from pgode.lib.latent_ode import LatentGraphODE
from pgode.lib.encoder_decoder import *
from pgode.lib.diffeq_solver import DiffeqSolver,GraphODEFunc
import torch.nn as nn



def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):


	# dim related
	latent_dim = args.latents # ode output dimension
	rec_dim = args.rec_dims
	input_dim = input_dim
	ode_dim = args.ode_dims #ode gcn dimension

	#encoder related

	encoder_z0 = GNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
						 aggregate=args.rec_attention, num_ball=args.n_balls)  # [b,n_ball,e]

	encoder_z0_g = GNN(in_dim=input_dim, n_hid=rec_dim, out_dim=args.latents_global, n_heads=args.n_heads,
					 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
					 aggregate=args.rec_attention, num_ball=args.n_balls, graph_level=True)  # [b,n_ball,e]


	#ODE related
	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim
	else:
		ode_input_dim = latent_dim


	ode_func_net = nn.ModuleList([GNN(in_dim=ode_input_dim, n_hid=ode_dim, out_dim=ode_input_dim, n_heads=args.n_heads,
						n_layers=args.gen_layers, dropout=args.dropout,
						conv_name=args.odenet, aggregate="add") for _ in range(args.expert_num)])

	gate_func_net = GateGNN(in_dim=ode_input_dim, n_hid=ode_dim, out_dim=ode_input_dim, obs_dim=input_dim, n_heads=args.n_heads,
							args=args, n_layers=args.gen_layers, dropout=args.dropout,
							conv_name=args.odenet, aggregate="add", expert_num=args.expert_num)

	# Decoder related
	decoder = Decoder(latent_dim, input_dim).to(device)
	print('latent_dim', latent_dim)

	gen_ode_func = GraphODEFunc(
		ode_func_net=ode_func_net,
		decoder=decoder,
		args=args,
		device=device).to(device)

	diffeq_solver = DiffeqSolver(gen_ode_func, gate_func_net, decoder, args.solver, args=args, odeint_rtol=1e-2,
								 odeint_atol=1e-2, device=device)


	model = LatentGraphODE(
		input_dim = input_dim,
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0,
		encoder_z0_g=encoder_z0_g,
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		MI_coef = args.MI_coef,
		args=args
	).to(device)

	return model
