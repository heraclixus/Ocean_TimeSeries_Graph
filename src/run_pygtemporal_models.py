from pygtemporal_models.pyg_temp_dataset import SSTDatasetLoader, inverse_normalize
import torch
import torch.optim as optim
import numpy as np
import argparse
from pygtemporal_models.math_utils import weighted_mse
from torch_geometric_temporal.nn.attention.mtgnn import MTGNN
from pygtemporal_models.stemgnn import StemGNN
from pygtemporal_models.agcrn import AGCRN
from pygtemporal_models.fouriergnn import FGN
from pygtemporal_models.graphwavenet import gwnet
from utils_pca import reconstruct_enso
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/sst_pcs.mat")
    parser.add_argument("--model_name", type=str, default="mtgnn")
    parser.add_argument("--multi_layer", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)
    parser.add_argument("--rnn_units", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64) # fgnn
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--use_normalization", action="store_true")
    parser.add_argument("--use_loss_weights", action="store_true")
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sst_dataloader = SSTDatasetLoader(filepath=args.input_file, use_normalization=args.use_normalization)

    print(f"max = {sst_dataloader._max.shape}, min = {sst_dataloader._min.shape}")

    train_dataset, test_dataset = sst_dataloader.get_dataset(window=args.window, horizon=args.horizon)
    train_input = np.array(train_dataset.features)
    train_target = np.array(train_dataset.targets) 
    print(f"train_input = {train_input.shape}")
    print(f"train_target = {train_target.shape}")
    test_input = np.array(test_dataset.features)
    test_target = np.array(test_dataset.targets) 
    print(f"test_input = {test_input.shape}")
    print(f"test_target = {test_target.shape}")


    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).unsqueeze(1).to(device)  # (B, F, N, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).unsqueeze(1).to(device)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=32, shuffle=False, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).unsqueeze(1).to(device) # (B, F, N, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).unsqueeze(1).to(device)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=32, shuffle=False,drop_last=True)

    print(f"train_x_tensor = {train_x_tensor.shape}")
    print(f'train_target_tensor = {train_target_tensor.shape}')
    print(f"test_x_tensor = {test_x_tensor.shape}")
    print(f"test_target_tensor = {test_target_tensor.shape}")


    # MTGNN configurations
    # build_adj = True means the graph construction layer generates adjancy matrix on the fly (sparse)
    # kernel_sizes are 1,3,6,7 in their casefor the temporal convolution dilated inception nets
    # subgraph size was for efficiency, we have a small graph, so same as graph size
    # seq_length is the window size for time series forecast
    # out_dim = 24 for the horizon size for time series forecast.

    if args.model_name == "mtgnn":
        model = MTGNN(gcn_true=True, build_adj=True, gcn_depth=3, num_nodes=sst_dataloader._n_nodes, 
                  kernel_set=[1,1,1,1], 
                  kernel_size=1, dropout=0.3, 
                  subgraph_size=sst_dataloader._n_nodes, 
                  node_dim=1, 
                  dilation_exponential=1,
                  conv_channels=32, 
                  residual_channels=32, 
                  skip_channels=64, 
                  end_channels=128, 
                  seq_length=12, 
                  in_dim=1, 
                  out_dim=24, 
                  layers=3, 
                  propalpha=0.05, 
                  tanhalpha=3, 
                  layer_norm_affline=True).to(device)
    elif args.model_name == "stemgnn":
        model = StemGNN(units=sst_dataloader._n_nodes,
                        stack_cnt=2,
                        time_step=args.window,
                        multi_layer=args.multi_layer,
                        horizon=args.horizon).to(device)
    elif args.model_name == "agcrn": 
        model = AGCRN(args=args, num_nodes=sst_dataloader._n_nodes).to(device)
    elif args.model_name == "fgnn":
        model = FGN(pre_length=args.horizon, 
                    embed_size=args.embed_dim, 
                    feature_size=sst_dataloader._n_nodes, 
                    seq_length=args.window, 
                    hidden_size=args.rnn_units).to(device)
    else: # wavenet
        model = gwnet(device=device, 
                      window=args.window, 
                      horizon=args.horizon,
                      num_nodes=sst_dataloader._n_nodes,
                      in_dim=args.input_dim,
                      out_dim=args.horizon).to(device)

    # encoder_input, label = next(iter(train_loader))
    # print(f'label = {label.shape}') 
    # print(f"encoder_input = {encoder_input.shape}")
    # if args.model_name == "stemgnn":
    #     output, attention = model(encoder_input)
    #     print(f"output = {output.shape}, attention = {attention.shape}")
    # else:
    #     output = model(encoder_input)
    #     print(f"output = {output.shape}")

    # obtain the adjacency matrix learned.
    # A_tide = model._graph_constructor(model._idx.to(device))
    # print(f"A_tide = {A_tide.shape}"
    # output2, attention2 = model(test_x_tensor)
    # print(f"output2 = {output2.shape}, attention2 = {attention2.shape}")

    # training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    total_param = 0
    for param_tensor in model.state_dict():
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    #--------------------------------------------------
    cumulative_patience = 0
    loss_fn = torch.nn.MSELoss()
    best_test_loss = np.inf

    losses_train, losses_test = [],[]
    rmses_train, rmses_test = [],[]

    rmses_train_reconstructed, rmses_test_reconstructed = [],[]

    for epoch in range(args.epochs):
        rmses_epoch = [] 
        rmses_reconstructed_epoch = []
        loss_list_epoch = []
        for i, (encoder_input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.model_name == "stemgnn":
                output, _ = model(encoder_input)
            else:
                # print(f"label = {label.shape}")
                output = model(encoder_input).permute(0,3,2,1)
                # print(f"output = {output.shape}")
                # exit(0)
            if args.use_loss_weights:
                loss = weighted_mse(label, output, sst_dataloader._std)
            else:
                loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            loss_list_epoch.append(loss.item())

            # compute rmse
            if args.use_normalization:
                label_np = inverse_normalize(label.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
                pred_np = inverse_normalize(output.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
            else:
                label_np = label.detach().cpu().numpy()
                pred_np = output.detach().cpu().numpy()
            rmse = np.sqrt(np.mean((label_np - pred_np)**2))
            rmses_epoch.append(rmse)
            # reconstructenso
            nino34, nino34_pred = reconstruct_enso(pred_np)
            rmse_reconstructed = np.sqrt(np.mean((nino34 - nino34_pred)**2))
            rmses_reconstructed_epoch.append(rmse_reconstructed)
        mean_loss_tr = np.mean(loss_list_epoch)
        mean_rmse_tr = np.mean(rmses_epoch)
        mean_rmse_reconstructed_tr = np.mean(rmses_reconstructed_epoch)
        print(f"Epoch {epoch} Train loss: {mean_loss_tr: .3f}, RMSE: {mean_rmse_tr :.3f}, RMSE Reconstructed: {mean_rmse_reconstructed_tr:.3f}")
        losses_train.append(mean_loss_tr)
        rmses_train.append(mean_rmse_tr) 
        rmses_train_reconstructed.append(mean_rmse_reconstructed_tr)

        # NOTE: 1/12, evaluate after every train to plot training dynamics
        model.eval()
        rmses_test_epoch = [] 
        losses_test_epoch = []
        rmses_test_reconstructed_epoch = []
        for i, (encoder_input, label) in enumerate(test_loader):
            if args.model_name == "stemgnn":
                output, _ = model(encoder_input)
            else:
                output = model(encoder_input).permute(0,3,2,1)
            # print(f"output = {output.shape}, label = {label.shape}")
            assert output.size() == label.size()
            if args.use_loss_weights:
                loss = weighted_mse(label, output, sst_dataloader._std)
            else:
                loss = loss_fn(output, label)
            losses_test_epoch.append(loss.item())
            # compute rmse
            if args.use_normalization:
                label_np = inverse_normalize(label.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
                pred_np = inverse_normalize(output.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
            else:
                label_np = label.detach().cpu().numpy()
                pred_np = output.detach().cpu().numpy()
            rmse = np.sqrt(np.mean((label_np - pred_np)**2))
            rmses_test_epoch.append(rmse)
            # reconstructenso
            nino34, nino34_pred = reconstruct_enso(pred_np)
            rmse_reconstructed = np.sqrt(np.mean((nino34 - nino34_pred)**2))
            rmses_test_reconstructed_epoch.append(rmse_reconstructed)
        test_rmse = np.mean(rmses_test_epoch)
        test_rmse_reconstructed = np.mean(rmses_test_reconstructed_epoch)
        test_epoch_loss = np.mean(losses_test_epoch)

        losses_test.append(test_epoch_loss)
        rmses_test.append(test_rmse)
        rmses_test_reconstructed.append(test_rmse_reconstructed)

        print(f"Epoch {epoch}, Test loss: {test_epoch_loss: .3f}, RMSE: {test_rmse: .3f}, RMSE Reconstructed: {test_rmse_reconstructed:.3f}")

        if test_epoch_loss <= best_test_loss:
            best_test_loss = test_epoch_loss
            cumulative_patience = 0
        else:
            cumulative_patience += 1 
        if cumulative_patience == args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        model.train()
    

    # save the final model's results
    # combine use_normalization and use_loss_weights to create save name
    save_name = f"{args.model_name}"
    if args.use_normalization:
        save_name += "_normalization"
    if args.use_loss_weights:
        save_name += "_weighted_loss"
    save_path = f"results/pytemporal/{save_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    # visualize train and test losses 
    fig_loss = plt.figure(figsize=(10,5))
    plt.plot(losses_train, label="Train Loss")
    plt.plot(losses_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train and Test Losses")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"losses.png"))
    plt.close() 

    # visualize train and test rmses
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

    # forecast again and save results
    if args.model_name == "stemgnn":
        output, attention = model(test_x_tensor)
        np.save(os.path.join(save_path, f"test_attention.npy"), attention.cpu().detach().numpy())
    else:
        output = model(test_x_tensor).permute(0,3,2,1)
    if args.use_normalization:
        output_np = inverse_normalize(output.cpu().detach().numpy(), sst_dataloader._max, sst_dataloader._min)
        test_target_np = inverse_normalize(test_target_tensor.cpu().detach().numpy(), sst_dataloader._max, sst_dataloader._min)
    else:
        output_np = output.cpu().detach().numpy()
        test_target_np = test_target_tensor.cpu().detach().numpy()  
    np.save(os.path.join(save_path, f"test_pred.npy"), output_np)
    np.save(os.path.join(save_path, f"test_true.npy"), test_target_np)
    if args.model_name == "mtgnn":
        A_tilde = model._graph_constructor(model._idx.to(device))
        np.save(os.path.join(save_path, f"A_tilde.npy"), A_tilde.cpu().detach().numpy())
    if args.model_name == "wavenet":
        A_tilde = model.new_supports[0]
        np.save(os.path.join(save_path, f"A_tilde.npy"), A_tilde.cpu().detach().numpy())