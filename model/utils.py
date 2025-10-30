import ast
import dgl
import torch
import time
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.nn import L1Loss, MSELoss


def get_encoded_tasks(list, offset=0):
    task_dict = {}
    encoded_tasks = []
    for task in list:
        if task not in task_dict:
            task_dict[task] = len(task_dict) + offset
        encoded_tasks.append(task_dict[task])
    return encoded_tasks, task_dict


def get_encoded_clusters(list, offset=0):
    cluster_dict = {}
    encoded_clusters = []
    for cluster in list:
        if cluster not in cluster_dict:
            cluster_dict[cluster] = len(cluster_dict) + offset
        encoded_clusters.append(cluster_dict[cluster])
    return encoded_clusters, cluster_dict


def create_in_features_node_dict(hetero_graph):
    in_features_dict_node = {}
    for node in hetero_graph.ntypes:
        in_features_dict_node[node] = hetero_graph.nodes[node].data['feat'].shape[1]
    return in_features_dict_node


def create_graph_edges_train(runtime_edges_data_train, add_reverse_edges=True):
    n_runtime_records = runtime_edges_data_train.shape[0]
    
    cluster_runtime_src_array, cluster_dict = get_encoded_clusters(
        runtime_edges_data_train['appMasterHost'].to_numpy()
    )
    runtime_task_src_array = list(range(n_runtime_records))
    runtime_task_dst_array, task_dict = get_encoded_tasks(
        runtime_edges_data_train['appId'].to_numpy()
    )
    
    graph_data = {
        ('cluster', 'cluster-runtime', 'runtime'): (cluster_runtime_src_array, runtime_task_src_array),
        ('runtime', 'runtime-task', 'task'): (runtime_task_src_array, runtime_task_dst_array)
    }
    
    if add_reverse_edges:
        graph_data[('runtime', 'cluster-runtime-reverse', 'cluster')] = (
            runtime_task_src_array, cluster_runtime_src_array
        )
        graph_data[('task', 'task-runtime', 'runtime')] = (
            runtime_task_dst_array, runtime_task_src_array
        )
    
    hetero_graph = dgl.heterograph(graph_data)
    return hetero_graph, cluster_dict, task_dict


def convert_runtime_to_seconds(runtime_str):
    if pd.isna(runtime_str):
        return 0.0
    
    parts = runtime_str.split()
    if len(parts) == 3:
        days = int(parts[0])
        time_str = parts[2]
    else:
        days = 0
        time_str = runtime_str
    
    time_parts = time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def create_train_graph(tasks_data, runtime_edges_data, clusters_data, task_features):
    runtime_values = runtime_edges_data['runtime'].apply(convert_runtime_to_seconds).values
    runtime_log = np.log1p(runtime_values).reshape(-1, 1)
    runtime_scaler = StandardScaler()
    runtime_normalized = runtime_scaler.fit_transform(runtime_log)
    ground_truth = torch.tensor(runtime_normalized).to(torch.float32)
    
    hetero_graph, cluster_dict, task_dict = create_graph_edges_train(runtime_edges_data)

    task_feat_tensor = torch.tensor(task_features.values).to(torch.float32)
    hetero_graph.nodes['task'].data['feat'] = nn.Parameter(task_feat_tensor, requires_grad=False)
    
    cluster_feat = clusters_data['feat'].apply(lambda x: ast.literal_eval(x))
    cluster_feat_df = pd.DataFrame(cluster_feat.tolist())
    
    cluster_feat_log = np.log1p(cluster_feat_df.values)
    cluster_scaler = StandardScaler()
    cluster_feat_normalized = cluster_scaler.fit_transform(cluster_feat_log)
    cluster_feat_tensor = torch.tensor(cluster_feat_normalized).to(torch.float32)
    hetero_graph.nodes['cluster'].data['feat'] = nn.Parameter(cluster_feat_tensor, requires_grad=False)
    runtime_to_task_list = []
    runtime_to_cluster_list = []
    
    for _, row in runtime_edges_data.iterrows():
        task_id = task_dict[row['appId']]
        cluster_id = cluster_dict[row['appMasterHost']]
        runtime_to_task_list.append(task_id)
        runtime_to_cluster_list.append(cluster_id)
    
    runtime_to_task_idx = torch.tensor(runtime_to_task_list, dtype=torch.long)
    runtime_to_cluster_idx = torch.tensor(runtime_to_cluster_list, dtype=torch.long)
    
    task_feat_for_runtime = task_feat_tensor[runtime_to_task_idx]
    cluster_feat_for_runtime = cluster_feat_tensor[runtime_to_cluster_idx]
    
    runtime_feat = torch.cat([task_feat_for_runtime, cluster_feat_for_runtime], dim=1)
    hetero_graph.nodes['runtime'].data['feat'] = nn.Parameter(runtime_feat, requires_grad=False)
    node_dict = {}
    edge_dict = {}
    for ntype in hetero_graph.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hetero_graph.etypes:
        edge_dict[etype] = len(edge_dict)
        hetero_graph.edges[etype].data["id"] = (
            torch.ones(hetero_graph.num_edges(etype), dtype=torch.long) * edge_dict[etype]
        )
    
    in_features_dict_node = create_in_features_node_dict(hetero_graph)
    
    return hetero_graph, ground_truth, node_dict, edge_dict, in_features_dict_node, runtime_scaler, cluster_scaler, task_dict, cluster_dict


def create_test_graph_edges(og_hetero_graph, runtime_edges_data_test):
    runtime_offset = og_hetero_graph.number_of_nodes('runtime')
    task_offset = og_hetero_graph.number_of_nodes('task')
    cluster_offset = og_hetero_graph.number_of_nodes('cluster')
    
    n_runtime_records = runtime_edges_data_test.shape[0]
    
    cluster_runtime_src_array, cluster_dict_test = get_encoded_clusters(
        runtime_edges_data_test['appMasterHost'].to_numpy(), cluster_offset
    )
    runtime_task_src_array = [i + runtime_offset for i in range(n_runtime_records)]
    runtime_task_dst_array, task_dict_test = get_encoded_tasks(
        runtime_edges_data_test['appId'].to_numpy(), task_offset
    )
    
    graph_data = {
        ('cluster', 'cluster-runtime', 'runtime'): (cluster_runtime_src_array, runtime_task_src_array),
        ('runtime', 'cluster-runtime-reverse', 'cluster'): (runtime_task_src_array, cluster_runtime_src_array),
        ('runtime', 'runtime-task', 'task'): (runtime_task_src_array, runtime_task_dst_array),
        ('task', 'task-runtime', 'runtime'): (runtime_task_dst_array, runtime_task_src_array)
    }
    
    return graph_data, cluster_dict_test, task_dict_test


def create_test_graph(hetero_graph, runtime_edges_data_test, task_labels, edge_dict, 
                     task_features, runtime_scaler, cluster_scaler, train_task_dict, train_cluster_dict):
    task_feat_tensor = torch.tensor(task_features.values).to(torch.float32)
    
    runtime_values = runtime_edges_data_test['runtime'].apply(convert_runtime_to_seconds).values
    runtime_log = np.log1p(runtime_values).reshape(-1, 1)
    runtime_normalized = runtime_scaler.transform(runtime_log)
    ground_truth_test = torch.tensor(runtime_normalized).to(torch.float32)
    
    test_edges, test_cluster_dict, test_task_dict = create_test_graph_edges(hetero_graph, runtime_edges_data_test)
    
    runtime_to_task_list = []
    runtime_to_cluster_list = []
    
    n_train_tasks = hetero_graph.number_of_nodes('task') - task_features.shape[0]
    n_train_clusters = hetero_graph.number_of_nodes('cluster')
    
    test_task_label_to_idx = {label: i + n_train_tasks for i, label in enumerate(task_labels)}
    
    for _, row in runtime_edges_data_test.iterrows():
        task_idx = test_task_label_to_idx[row['appId']]
        runtime_to_task_list.append(task_idx)
        
        cluster_name = row['appMasterHost']
        if cluster_name in train_cluster_dict:
            cluster_idx = train_cluster_dict[cluster_name]
        else:
            cluster_idx = 0
        runtime_to_cluster_list.append(cluster_idx)
    
    runtime_to_task_idx = torch.tensor(runtime_to_task_list, dtype=torch.long)
    runtime_to_cluster_idx = torch.tensor(runtime_to_cluster_list, dtype=torch.long)
    
    merged_task_feat = torch.cat((hetero_graph.nodes['task'].data['feat'], task_feat_tensor), 0)
    merged_task_feat = nn.Parameter(merged_task_feat, requires_grad=False)
    
    task_feat_for_test_runtime = merged_task_feat[runtime_to_task_idx]
    cluster_feat_for_test_runtime = hetero_graph.nodes['cluster'].data['feat'][runtime_to_cluster_idx]
    
    runtime_feat_test = torch.cat([task_feat_for_test_runtime, cluster_feat_for_test_runtime], dim=1)
    runtime_feat_test = nn.Parameter(runtime_feat_test, requires_grad=False)
    
    merged_runtime_feat = torch.cat((hetero_graph.nodes['runtime'].data['feat'], runtime_feat_test), 0)
    merged_runtime_feat = nn.Parameter(merged_runtime_feat, requires_grad=False)
    
    edge_ids = {}
    for etype in hetero_graph.etypes:
        edge_ids[etype] = hetero_graph.edges[etype].data['id']
    
    num_new_tasks = task_features.shape[0]
    num_new_runtime = runtime_edges_data_test.shape[0]
    hetero_graph.add_nodes(num_new_tasks, ntype='task')
    hetero_graph.add_nodes(num_new_runtime, ntype='runtime')
    
    for edge_type in test_edges:
        arr_src, arr_dst = test_edges[edge_type]
        hetero_graph.add_edges(arr_src, arr_dst, etype=edge_type[1])
        tt = torch.ones(len(arr_src), dtype=torch.long) * edge_dict[edge_type[1]]
        edge_ids[edge_type[1]] = torch.cat((edge_ids[edge_type[1]], tt), 0)
        hetero_graph.edges[edge_type[1]].data["id"] = edge_ids[edge_type[1]]
    
    hetero_graph.nodes['task'].data['feat'] = merged_task_feat
    hetero_graph.nodes['runtime'].data['feat'] = merged_runtime_feat
    
    return hetero_graph, ground_truth_test


def prepare_splits(tasks_data, runtime_data, train_mask, val_mask, test_mask, normalize_flag=True):
    tasks_train_f = tasks_data[train_mask]
    tasks_val_f = tasks_data[val_mask]
    tasks_test_f = tasks_data[test_mask]
    
    task_instances_train_f = tasks_train_f['label'].values
    task_instances_val_f = tasks_val_f['label'].values
    task_instances_test_f = tasks_test_f['label'].values
    
    runtime_train_f = runtime_data[runtime_data['appId'].isin(task_instances_train_f)]
    runtime_val_f = runtime_data[runtime_data['appId'].isin(task_instances_val_f)]
    runtime_test_f = runtime_data[runtime_data['appId'].isin(task_instances_test_f)]
    
    task_feat_train_f = tasks_train_f['feat'].apply(ast.literal_eval).apply(pd.Series)
    task_feat_val_f = tasks_val_f['feat'].apply(ast.literal_eval).apply(pd.Series)
    task_feat_test_f = tasks_test_f['feat'].apply(ast.literal_eval).apply(pd.Series)
    
    if normalize_flag:
        task_feat_train_log = np.log1p(task_feat_train_f.values)
        task_feat_val_log = np.log1p(task_feat_val_f.values)
        task_feat_test_log = np.log1p(task_feat_test_f.values)
        
        scaler = StandardScaler()
        task_feat_train_normalized = scaler.fit_transform(task_feat_train_log)
        task_feat_val_normalized = scaler.transform(task_feat_val_log)
        task_feat_test_normalized = scaler.transform(task_feat_test_log)
        
        task_feat_train_f = pd.DataFrame(
            task_feat_train_normalized, 
            index=task_feat_train_f.index,
            columns=task_feat_train_f.columns
        )
        task_feat_val_f = pd.DataFrame(
            task_feat_val_normalized, 
            index=task_feat_val_f.index,
            columns=task_feat_val_f.columns
        )
        task_feat_test_f = pd.DataFrame(
            task_feat_test_normalized, 
            index=task_feat_test_f.index,
            columns=task_feat_test_f.columns
        )
        
    
    return (tasks_train_f, tasks_val_f, tasks_test_f, 
            runtime_train_f, runtime_val_f, runtime_test_f, 
            task_feat_train_f, task_feat_val_f, task_feat_test_f)


def train_inductive(hetero_graph, hetero_graph_val, model, optimizer, scheduler, n_epochs, 
                    ground_truth_train, ground_truth_val, model_loc_save, seed, 
                    return_predictions=False, clip_grad_norm=0.5,device='cpu'):
    set_random_seed(seed)
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_train_mse = float('inf')
    best_val_mse = float('inf')
    best_train_l1 = float('inf')
    best_val_l1 = float('inf')
    best_epoch = -1
    best_pred = None
    save_lr = 0
    
    start_time = time.time()
    for epoch in range(n_epochs):
        model.train()
        embeddings = model(hetero_graph, hetero_graph.ndata['feat'], out_key='none')
        predictions = model.predict(embeddings)
        
        loss_fn = L1Loss()
        mse_fn = MSELoss()
        l1_fn = L1Loss()
        
        train_loss = loss_fn(predictions, ground_truth_train)
        train_mse = mse_fn(predictions, ground_truth_train)
        train_l1 = l1_fn(predictions, ground_truth_train)
        
        model.eval()
        with torch.no_grad():
            embeddings_val = model(hetero_graph_val, hetero_graph_val.ndata['feat'], out_key='none')
            predictions_val = model.predict(embeddings_val)
        
        val_mask = [False if i < predictions.shape[0] else True for i in range(predictions_val.shape[0])]
        predictions_val = predictions_val[val_mask]
        model.train()
        
        val_loss = loss_fn(predictions_val, ground_truth_val)
        val_mse = mse_fn(predictions_val, ground_truth_val)
        val_l1 = l1_fn(predictions_val, ground_truth_val)
        
        optimizer.zero_grad()
        train_loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:3d} | Train L1: {train_l1.item():.4f} | Val L1: {val_l1.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_loc_save)
            best_pred = predictions_val
            best_train_l1 = train_l1
            best_val_l1 = val_l1
            best_train_mse = train_mse
            best_val_mse = val_mse
            training_time = (time.time() - start_time) * 1000
            save_lr = optimizer.param_groups[0]['lr']
    
    if return_predictions:
        return (model, best_epoch, save_lr, best_train_l1, best_train_mse, 
                best_val_l1, best_val_mse, training_time, best_pred)
    else:
        return (model, best_epoch, save_lr, best_train_l1, best_train_mse, 
                best_val_l1, best_val_mse, training_time)


def inference(model, model_loc_save, graph_test, y_true_test, seed,device='cpu'):
    set_random_seed(seed)
    model.load_state_dict(torch.load(model_loc_save))
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        embeddings_test = model(graph_test, graph_test.ndata['feat'], out_key='none')
        y_pred_test = model.predict(embeddings_test, out_key='runtime')
    
    inference_time = (time.time() - start_time) * 1000
    num_train_examples = len(y_pred_test) - len(y_true_test)
    y_pred_test = y_pred_test[num_train_examples:]
    
    l1_loss_fn = L1Loss()
    mse_loss_fn = MSELoss()
    l1_test_loss = l1_loss_fn(y_pred_test, y_true_test)
    mse_test_loss = mse_loss_fn(y_pred_test, y_true_test)
    
    return y_pred_test, l1_test_loss, mse_test_loss, inference_time


def read_input_data(loc):
    clusters_data = pd.read_csv(f'{loc}/datasets/clusters.csv')
    runtime_data = pd.read_csv(f'{loc}/datasets/runtimes.csv')
    tasks_data = pd.read_csv(f'{loc}/datasets/tasks.csv')
    return clusters_data, runtime_data, tasks_data


def set_random_seed(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_splits(task_instances, outer_fold):
    index = task_instances.index.tolist()
    task_instances = task_instances.to_numpy()
    train_mask = np.zeros(len(task_instances), dtype=bool)
    val_mask = np.zeros(len(task_instances), dtype=bool)
    test_mask = np.zeros(len(task_instances), dtype=bool)
    
    test_iid = outer_fold
    range_without_test_iid = [i for i in range(1, 6) if i != test_iid]
    val_iid = np.random.choice(range_without_test_iid, 1)[0]
    
    n_tasks = len(task_instances)
    indices = np.arange(n_tasks)
    np.random.shuffle(indices)
    
    test_size = n_tasks // 5
    val_size = n_tasks // 5
    
    test_start = (test_iid - 1) * test_size
    test_indices = indices[test_start:test_start + test_size]
    val_indices = indices[(val_iid - 1) * val_size:(val_iid - 1) * val_size + val_size]
    
    train_mask[indices] = True
    train_mask[test_indices] = False
    train_mask[val_indices] = False
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask, val_iid