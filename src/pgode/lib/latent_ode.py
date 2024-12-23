from .base_models import VAE_Baseline
import pgode.lib.utils as utils
import torch
import torch.nn as nn
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation


class FF(nn.Module):
	def __init__(self, input_dim, out_dim):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(input_dim, out_dim),
			nn.ReLU(),
			nn.Linear(out_dim, out_dim),
			nn.ReLU(),
			nn.Linear(out_dim, out_dim),
			nn.ReLU()
		)
		self.linear_shortcut = nn.Linear(input_dim, out_dim)
		self.init_emb()

	def init_emb(self):
		# initrange = -1.5 / self.embedding_dim
		for m in self.modules():
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)
	def forward(self, x):
		return self.block(x) + self.linear_shortcut(x)


class LatentGraphODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, encoder_z0_g, decoder, diffeq_solver,
				 z0_prior, device, args, obsrv_std=None, MI_coef=0.1):

		super(LatentGraphODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std)

		self.encoder_z0 = encoder_z0
		self.encoder_z0_g = encoder_z0_g
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.latent_dim =latent_dim
		self.MI_coef = MI_coef
		self.num_atoms = args.n_balls
		self.save_traj = args.save_traj
		self.args = args

		self.local_d = FF(args.latents * args.n_balls, args.latents)
		self.global_d = FF(args.latents_global, args.latents)
		self.sys_d = FF(args.num_sys_paras, args.latents)

	def compute_mutual_information(self, emb1, emb2, measure='JSD'):
		assert emb1.shape[0] == emb2.shape[0]
		num_graphs = emb1.shape[0]

		with torch.no_grad():
			pos_mask = torch.zeros((num_graphs, num_graphs), dtype=torch.bool).to(emb1.device)
			pos_mask.fill_diagonal_(1)
			neg_mask = ~pos_mask

		emb1 = torch.nn.functional.normalize(emb1, dim=-1)
		emb2 = torch.nn.functional.normalize(emb2, dim=-1)
		res = torch.mm(emb1, emb2.t())

		E_pos = get_positive_expectation(res * pos_mask, measure, average=True)
		E_neg = get_negative_expectation(res * neg_mask, measure, average=True)

		return E_pos - E_neg

	def get_reconstruction(self, batch_en,batch_de, batch_g,n_traj_samples=1,run_backwards=True):

        #Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_attr,
														  batch_en.edge_index, batch_en.pos, batch_en.edge_same,
														  batch_en.batch, batch_en.y)  # [num_ball,16]

		first_point_g = self.encoder_z0_g(batch_en.x, batch_en.edge_attr,
											 batch_en.edge_index, batch_en.pos, batch_en.edge_same,
											 batch_en.batch, batch_en.y)  # [num_ball//10,16]

		first_point_l = first_point_mu
		n_traj, feature = first_point_l.size()[0], first_point_l.size()[1]
		batch_size, feature_g = first_point_g.size()[0], first_point_g.size()[1]
		assert batch_size * self.num_atoms == n_traj

		first_point_l_emb = self.local_d(first_point_l.reshape(batch_size, self.num_atoms * feature))
		first_point_g_emb = self.global_d(first_point_g)
		disen_loss = self.compute_mutual_information(first_point_l_emb, first_point_g_emb)

		means_z0 = first_point_mu.repeat(n_traj_samples,1,1) #[3,num_ball,16], num_ball=B*N=256*5=1280
		sigmas_z0 = first_point_std.repeat(n_traj_samples,1,1) #[3,num_ball,16]
		first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0) #[3,num_ball,16]

		first_point_std = first_point_std.abs()


		time_steps_to_predict = batch_de["time_steps"]


		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())
		assert (not torch.isnan(first_point_enc).any())


		# ODE:Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		if self.args.gate_use_global:
			sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g,
									   first_point_g=first_point_g.repeat(n_traj_samples,1,1))
		else:
			sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)

        # Decoder:
		pred_x = self.decoder(sol_y)


		all_extra_info = {
			"first_point": (torch.unsqueeze(first_point_mu,0), torch.unsqueeze(first_point_std,0), first_point_enc),
			# "first_point_global": (torch.unsqueeze(first_point_mu_g, 0), torch.unsqueeze(first_point_std_g, 0),
			# 					   first_point_enc_g),
			"latent_traj": sol_y.detach(),
			"disen_loss": disen_loss
		}

		return pred_x, all_extra_info, None



