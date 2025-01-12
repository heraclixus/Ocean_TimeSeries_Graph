from pygtemporal_models.pyg_temp_dataset import SSTDatasetLoader, inverse_normalize
import torch
import torch.optim as optim
import numpy as np
import argparse
from torch_geometric_temporal.nn.attention.mtgnn import MTGNN
from pygtemporal_models.stemgnn import StemGNN
from pygtemporal_models.agcrn import AGCRN
from pygtemporal_models.fouriergnn import FGN
from pygtemporal_models.graphwavenet import gwnet
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/sst_pcs.mat")
    parser.add_argument("--output_file", type=str, default="../data/sst_pcs_mtgnn.mat")
    parser.add_argument("--model_name", type=str, default="mtgnn")
    parser.add_argument("--multi_layer", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)
    parser.add_argument("--rnn_units", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128) # fgnn
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sst_dataloader = SSTDatasetLoader(filepath=args.input_file)

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
    else:
        model = gwnet(device=device, 
                      num_nodes=sst_dataloader._n_nodes,
                      in_dim=args.input_dim,
                      out_dim=args.horizon).to(device)

    # encoder_input, label = next(iter(train_loader))
    # print(f'label = {label.shape}') 
    # print(f"encoder_input = {encoder_input.shape}")

    # output, attention = model(encoder_input)
    # print(f"output = {output.shape}, attention = {attention.shape}")

    # # obtain the adjacency matrix learned.
    # # A_tide = model._graph_constructor(model._idx.to(device))
    # # print(f"A_tide = {A_tide.shape}"
    # output2, attention2 = model(test_x_tensor)
    # print(f"output2 = {output2.shape}, attention2 = {attention2.shape}")

    # exit(0)

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

    for epoch in range(args.epochs):
        rmses = [] 
        loss_list = []
        for i, (encoder_input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.model_name == "stemgnn":
                output, _ = model(encoder_input)
                output = output.permute(0,3,2,1)
            else:
                # print(f"label = {label.shape}")
                output = model(encoder_input).permute(0,3,2,1)
                # print(f"output = {output.shape}")
                # exit(0)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            # compute rmse
            label_np = inverse_normalize(label.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
            pred_np = inverse_normalize(output.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
            rmse = np.sqrt(np.mean((label_np - pred_np)**2))
            rmses.append(rmse)
        print(f"Epoch {epoch} Train loss: {np.mean(loss_list)}, RMSE: {np.mean(rmses)}") 

        if epoch % args.eval_freq == 0 and epoch != 0:
            print(f"Evaluating model at epoch {epoch}")
            model.eval()
            rmses_test = [] 
            with torch.no_grad():
                loss_list = []
                for i, (encoder_input, label) in enumerate(test_loader):
                    if args.model_name == "stemgnn":
                        output, _ = model(encoder_input)
                        output = output.permute(0,3,2,1)
                    else:
                        output = model(encoder_input).permute(0,3,2,1)
                    loss_list.append(np.sqrt(loss.item()))

                    # compute rmse
                    label_np = inverse_normalize(label.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
                    pred_np = inverse_normalize(output.detach().cpu().numpy(), sst_dataloader._max, sst_dataloader._min)
                    rmse = np.sqrt(np.mean((label_np - pred_np)**2))
                    rmses_test.append(rmse)
                test_rmse = np.mean(rmses_test)
                test_epoch_loss = np.mean(loss_list)
                print(f"Epoch {epoch}, Test loss: {test_epoch_loss}, RMSE: {test_rmse}")

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
    save_path = f"results/pytemporal/{args.model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    # forecast again and save results
    if args.model_name == "stemgnn":
        output, attention = model(test_x_tensor)
        output = output.permute(0,3,2,1)
        np.save(os.path.join(save_path, f"test_attention.npy"), attention.cpu().detach().numpy())
    else:
        output = model(test_x_tensor).permute(0,3,2,1)
    np.save(os.path.join(save_path, f"test_pred.npy"), output.cpu().detach().numpy())
    np.save(os.path.join(save_path, f"test_true.npy"), label.cpu().detach().numpy())
    if args.model_name == "mtgnn":
        A_tilde = model._graph_constructor(model._idx.to(device))
        np.save(os.path.join(save_path, f"A_tilde.npy"), A_tilde.cpu().detach().numpy())
    if args.model_name == "wavenet":
        A_tilde = model.new_supports[0]
        np.save(os.path.join(save_path, f"A_tilde.npy"), A_tilde.cpu().detach().numpy())