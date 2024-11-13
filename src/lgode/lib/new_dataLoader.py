import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lgode.lib.utils as utils
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

"""
['pna', 'pacwarmpool', 'pacwarm', 'qbo', 'amon', 'ea', 
'nina1', 'wp', 'nao', 'soi', 'ammsst', 'nina3', 'tsa', 
'nina4', 'gmsst', 'tna', 'whwp', 'epo', 'solar', 'nina34', 
'tni', 'noi']
"""
category_1_features = ["nina1", "nina3", "nina34", "nina4"]
category_2_features = category_1_features + ["pacwarm", 'soi', "tni", "whwp"]
category_3_features = category_2_features + ["ammsst", "tna", "tsa", 'amon']
category_4_features = category_3_features + ["ea", "epo", "nao", "pna", "wp"]
category_5_features = category_4_features + ["qbo", "solar"]

 
def normalize(series, original_max, original_min):
    return (series - original_min) / (original_max - original_min)
def inverse_normalize(scaled_series, original_max, original_min):
    return scaled_series * (original_max-original_min) + original_min

# def normalize(series, original_max, original_min):
#     return (series - original_max) / original_min
# def inverse_normalize(scaled_series, original_max, original_min):
#     return scaled_series * original_min + original_max

 
class ParseData(object):

    def __init__(self, dataset_path,args,suffix='_springs5',mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.cutting_edge = args.cutting_edge
        self.num_pre = args.pred_len
        self.args = args
        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None
        self.feature_set = self.args.feature_set

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
  
    def preprocess(self, data_type):
        '''
        Output:
            timeseries: [#seq, #node, #timestep, #feature] (#feature=1)
            edges: [#seq, #node, #node]
            times: [#seq, #node, #timesteps] 
        ''' 
        print('--> Generating graphs and time series')
        time_series = []
        edges = []
        time_obs = [] # observed timestamps (required for LGODE)

        # ../data/indices_ocean_19_timeseries.csv
        data_np = np.loadtxt(self.args.input_file, delimiter=',', dtype=str, skiprows=1)
        data = pd.read_csv(self.args.input_file)

        if self.args.feature_set == 1:
            data = data[category_1_features].to_numpy()
            self.features = category_1_features
        elif self.args.feature_set == 2:
            data = data[category_2_features].to_numpy()
            self.features = category_2_features
        elif self.args.feature_set == 3:
            data = data[category_3_features].to_numpy()
            self.features = category_3_features
        elif self.args.feature_set == 4:
            data = data[category_4_features].to_numpy()
            self.features = category_4_features
        else:
            data =data[category_5_features].to_numpy() 
            self.features = category_5_features
        
        self.nino_feature_index = self.features.index("nina34")

        year_month = data_np[:, 0]                
        X = data
        ###### Split train and test 
        for i in range(len(year_month)):
            if int(year_month[i][:4]) == 2010:
                train_id = i - 1 
                break  
        if data_type == 'train':
            X = X[:train_id]
        else:
            X = X[train_id:]
        
        if self.args.test == 1:
            X = X[:72]
        
        T, N = X.shape
        ###### Chunk into shorter time series 
        seq_len = self.args.cond_len + self.args.pred_len  
        for i in range(T-seq_len):
            subseq = X[i:i+seq_len] # T' x N 
            subseq = subseq.T # N x T'
            subseq = subseq.reshape(N, seq_len, 1)
            time_series.append(subseq)
            edges.append(np.ones((N,N)))
            time_obs.append(np.tile(np.linspace(0,5,seq_len), (N,1)).reshape(N,-1))
         
        time_series = np.stack(time_series)  # train: n_seq x N x T x 1
        edges = np.stack(edges)
        time_obs = np.stack(time_obs)  
        
        
        original_max =  np.max(time_series, 2, keepdims=True)
        original_min =  np.min(time_series, 2, keepdims=True) 

        time_series = normalize(time_series, original_max, original_min)
        print('\ntime series max min', np.max(time_series), np.min(time_series)) 
        return time_series, edges, time_obs, original_max, original_min

    def load_data(self,sample_percent,batch_size,data_type="train"):
        self.batch_size = batch_size
        self.sample_percent = sample_percent  
        timeseries, edges, times, original_max, original_min = self.preprocess(data_type)  
        self.num_graph = timeseries.shape[0]
        self.num_atoms = timeseries.shape[1]
        self.args.n_balls = timeseries.shape[1]
        self.feature = timeseries.shape[-1]
        print("# graph in   " + data_type + "   is %d" % self.num_graph)
        print("# nodes in   " + data_type + "   is %d" % self.num_atoms)
 
         
        self.timelength = timeseries.shape[2]
  
        timeseries_en = timeseries[:,:,:self.args.cond_len]
        times_en = times[:,:,:self.args.cond_len]
        timeseries_de = timeseries[:,:,self.args.cond_len:]
        times_de = times[:,:,self.args.cond_len:]
        #Encoder dataloader
        series_list_observed, timeseries_observed, times_observed = self.split_data(timeseries_en, times_en)
        if self.mode == "interp":
            time_begin = 0
        else:
            time_begin = 1  
        
        print(f'\n############# verify shapes in {data_type}')
        print('Encoder feature: (#seq, #nodes, #obs_timestep, #feat) =', timeseries_observed.shape)
        
        self.args.std = np.mean([timeseries_observed[i,:,0,:].std() for i in range(self.num_graph)])
        
        print('Decoder feature: (#seq, #nodes, #pred_timestep, #feat) =', timeseries_de.shape)
        print('Encoder time: (#seq, #nodes, #obs_timestep) =', times_observed.shape)
        print('Decoder time: (#seq, #nodes, #pred_timestep) =', times_de.shape)
        print('edges: (#seq, #nodes, #nodes) =', edges.shape, '\n')
         
        encoder_data_loader, graph_data_loader = self.transfer_data(timeseries_observed, edges,
                                                                    times_observed, time_begin=time_begin)
 
        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size) 
 
        # Decoder Dataloader
        if self.mode=="interp":
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            series_list_de = self.decoder_data(timeseries_de,times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]
 
        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)
         
        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch, original_max , original_min
 
 
    def split_data(self,timeseries,times):
        # timeseries_observed = np.ones_like(timeseries) 
        # times_observed = np.ones_like(times)
        n_seq, n_node, total_time, n_feat = timeseries.shape   
        timeseries_observed = np.ones((n_seq, n_node, int(total_time * self.sample_percent), n_feat)) 
        times_observed = np.ones((n_seq, n_node, int(total_time * self.sample_percent)))

        # split encoder data
        timeseries_list = [] 
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                timeseries_list.append(timeseries[i][j]  )  # [2500] num_train * num_ball 
                times_list.append(times[i][j]  )

 
        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(timeseries_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series) 
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            timeseries_observed[graph_index][atom_index] = loc_series[preserved_idx] 
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]

            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = loc_series[preserved_idx]
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))

            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = loc_series
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)
             
            series_list.append((tt, vals, masks)) 
        
        return series_list, timeseries_observed, times_observed

    def decoder_data(self, time_series, times):

        # split decoder data
        loc_list = [] 
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(time_series[i][j])  # [2500] num_train * num_ball 
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = loc_series
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list
 
    def transfer_data(self,time_series, edges, times, time_begin=0):
        data_list = []
        graph_list = []
        edge_size_list = []

        for i in tqdm(range(self.num_graph)):
            data_per_graph, edge_data, edge_size = self.transfer_one_graph(time_series[i], edges[i], times[i],
                                                                           time_begin=time_begin)
            data_list.append(data_per_graph)
            graph_list.append(edge_data)
            edge_size_list.append(edge_size)

        print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=self.batch_size)
        graph_loader = DataLoader(graph_list, batch_size=self.batch_size)

        return data_loader, graph_loader

    def transfer_one_graph(self,time_series, edge, time, time_begin=0, mask=True, forward=False):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creeating pos 【N】
        # forward: t0=0;  otherwise: t0=tN/2 
        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix == "_springs5" or self.suffix == "_charged5":
                max_gap = (self.total_step - 40 * self.sample_percent) /self.total_step
            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100


        if self.mode=="interp":
            forward= False
        else:
            forward=True


        y = np.zeros(self.num_atoms)
        x = list()
        x_pos = list()
        node_number = 0
        node_time = dict()
        ball_nodes = dict()

        # Creating x, y, x_pos
        for i in range(len(time_series)):  
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball)

            # Creating x and x_pos, by tranverse each ball's sequence
            for j in range(time_series[i].shape[0]):
                xj_feature = time_series[i][j]
                x.append(xj_feature)

                x_pos.append(time_ball[j] - time_begin)
                node_time[node_number] = time_ball[j]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1

        '''
         matrix computing
         '''
        # Adding self-loop
        edge_with_self_loop = edge + np.eye(self.num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0)
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos)))

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if edge_with_self_loop[i][j] == 1:
                    sender_index_start = int(np.sum(y[:i]))
                    sender_index_end = int(sender_index_start + y[i])
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        if mask == None:
            edge_time_matrix = np.where(abs(edge_time_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are thosewhose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are thosewhose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        edge_index, edge_attr = self.convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge)



        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x = torch.ones(self.num_atoms),edge_index = graph_index_original)
 
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        edge_size = edge_index.shape[1]

        return graph_data,edge_data,edge_size

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True) #【including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])


        for b, ( tt, vals, mask) in enumerate(batch):

            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()
        
         

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
            }
        return data_dict

    def normalize_features(self,inputs, num_balls):
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr