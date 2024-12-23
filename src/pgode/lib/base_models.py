from pgode.lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch
import pickle
EPS=1e-12


class VAE_Baseline(nn.Module):
	def __init__(self, input_dim, latent_dim, 
		z0_prior, device,
		obsrv_std = 0.01, 
		):

		super(VAE_Baseline, self).__init__()
		
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior

	def get_gaussian_likelihood(self, truth, pred_y,temporal_weights, mask ):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask = mask.repeat(pred_y.size(0), 1, 1, 1)
		log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
			obsrv_std = self.obsrv_std, mask = mask,temporal_weights= temporal_weights) #【num_traj,num_sample_traj] [250,3]
		log_density_data = log_density_data.permute(1,0)
		log_density = torch.mean(log_density_data, 1)

		# shape: [n_traj_samples]
		return log_density


	def get_mse(self, truth, pred_y, mask = None, save_traj=False):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
		# shape: [1]

		mse = torch.mean(log_density_data)

		if self.save_traj and save_traj:
			with open('output/goat_trajs_test_{:4f}.pkl'.format(mse), 'wb') as fo:
				pickle.dump([pred_y.detach().cpu().numpy(), truth_repeated.detach().cpu().numpy(),
							 mask.detach().cpu().numpy(), mse.detach().cpu().numpy()], fo)
				fo.close()
			print('output/goat_trajs_test_{:4f}.pkl'.format(mse))
		# exit()

		return mse


	def compute_all_losses(self, batch_dict_encoder,batch_dict_decoder,batch_dict_graph,n_traj_samples=1,
						   kl_coef=1.):
		# Condition on subsampled points
		# Make predictions for all the points

		pred_y, info,temporal_weights= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder,batch_dict_graph,
															   n_traj_samples = n_traj_samples)
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]

		actual_y = batch_dict_decoder["data"]
		#print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs() + EPS
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")


		kldiv_z0 = torch.mean(kldiv_z0,(1,2))


		# Compute likelihood of all the points
		rec_likelihood = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_y,temporal_weights,
			mask=batch_dict_decoder["mask"])   #negative value

		# Compute the Fourier loss to see how periodicity goes in
		fft_true = torch.fft.fft(actual_y.squeeze(), dim=-1)
		fft_pred = torch.fft.fft(pred_y.squeeze(), dim=-1)
		fourier_loss = torch.mean(torch.abs(fft_true - fft_pred))

		mse = self.get_mse(
			batch_dict_decoder["data"], pred_y,
			mask=batch_dict_decoder["mask"], save_traj=True)  # [1]

		q_mse = self.get_mse(
			batch_dict_decoder["data"][:, :, :2], pred_y[:, :, :, :2],
			mask=batch_dict_decoder["mask"][:, :, :2])  # [1]

		v_mse = self.get_mse(
			batch_dict_decoder["data"][:, :, 2:], pred_y[:, :, :, 2:],
			mask=batch_dict_decoder["mask"][:, :, 2:])  # [1]

		feature_num = pred_y.shape[-1]
		mse_list = []
		for i in range(feature_num):
			mse_list.append(self.get_mse(
				batch_dict_decoder["data"][:, :, i:i + 1], pred_y[:, :, :, i:i + 1],
				mask=batch_dict_decoder["mask"][:, :, i:i + 1]))  # [1]

		# loss
		disen_loss = info["disen_loss"]
		sys_loss = info["sys_loss"]


		loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0) + self.args.disen_coef * disen_loss + self.MI_coef * sys_loss
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0) + self.args.disen_coef * disen_loss + self.MI_coef * sys_loss

		print(f"loss = {loss}")
		print(f"fourier_loss = {fourier_loss}")
		loss = loss + self.fourier_coeff * fourier_loss
 

		results = {}
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(rec_likelihood).data.item()
		results["mse"] = torch.mean(mse).data.item()
		results["kl_first_p"] =  torch.mean(kldiv_z0).detach().data.item()
		results["std_first_p"] = torch.mean(fp_std).detach().data.item()
		results["disen_loss"] = disen_loss.item() * self.MI_coef
		results["sys_loss"] = sys_loss.item() * self.MI_coef
		results["q_mse"] = torch.mean(q_mse).data.item()
		results["v_mse"] = torch.mean(v_mse).data.item()
		results["fourier_loss"] = fourier_loss.data.item()

		for i in range(feature_num):
			results["feature{}_mse".format(i)] = torch.mean(mse_list[i]).data.item()

		return results,








