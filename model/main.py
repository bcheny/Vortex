import torch
import pandas as pd
import os
import sys
import numpy as np
from GNN_architecture import HeteroGraphSage
from utils import (
    set_random_seed, read_input_data, prepare_splits, create_train_graph,
    create_test_graph, train_inductive, inference, create_data_splits
)


def train_predict(seed, n_hid, n_layers, dropout):
    loc = '.'
    out_loc = './results'
    set_random_seed(seed)

    n_epochs = 100
    learning_rate = 0.001
    weight_decay = 0.001
    patience = 20
    factor = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clusters_data, runtime_data, tasks_data = read_input_data(loc)
    
    all_predictions = []

    for outer_fold in range(1, 6):
        
        train_mask, val_mask, test_mask, val_iid = create_data_splits(tasks_data['label'], outer_fold)
        tasks_train, tasks_val, tasks_test, runtime_train, runtime_val, runtime_test, task_feat_train, task_feat_val, task_feat_test = prepare_splits(
            tasks_data, runtime_data, train_mask, val_mask, test_mask
        )

        graph, y_true_train, node_dict, edge_dict, in_features, runtime_scaler, cluster_scaler, train_task_dict, train_cluster_dict = create_train_graph(
            tasks_train, runtime_train, clusters_data, task_feat_train
        )
        
        graph_val, y_true_val = create_test_graph(
            graph.clone(), runtime_val, list(tasks_val['label']), edge_dict,
            task_feat_val, runtime_scaler, cluster_scaler, train_task_dict, train_cluster_dict
        )
        graph_test, y_true_test = create_test_graph(
            graph.clone(), runtime_test, list(tasks_test['label']), edge_dict,
            task_feat_test, runtime_scaler, cluster_scaler, train_task_dict, train_cluster_dict
        )

        graph = graph.to(device)
        graph_val = graph_val.to(device)
        graph_test = graph_test.to(device)
        y_true_train = y_true_train.to(device)
        y_true_val = y_true_val.to(device)
        y_true_test = y_true_test.to(device)


        os.makedirs(f'{out_loc}/models/', exist_ok=True)
        os.makedirs(f'{out_loc}/runtime/', exist_ok=True)
        os.makedirs(f'{out_loc}/predictions/', exist_ok=True)

        model_loc_save = f'{out_loc}/models/model_state_dict_{seed}_{n_hid}_{n_layers}_{dropout}.pth'

        model = HeteroGraphSage(n_layers, in_features, n_hid, 1, graph.etypes, dropout)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', patience=patience, factor=factor, verbose=False, min_lr=1e-5
        )
        

        trained_model, saved_epoch, saved_lr, l1_train, mse_train, l1_val, mse_val, training_time = train_inductive(
            graph, graph_val, model, opt, scheduler, n_epochs, y_true_train, y_true_val, 
            model_loc_save, seed, return_predictions=False,device=device
        )
        

        y_pred_test, l1_test, mse_test, inference_time = inference(
            model, model_loc_save, graph_test, y_true_test, seed,device=device
        )
        
        all_predictions.append({
            "outer_fold": outer_fold,
            "val_iid": val_iid,
            "y_pred_test": y_pred_test.cpu().numpy(),
            "y_true_test": y_true_test.cpu().numpy(),
            "l1_test": l1_test.item(),
            "mse_test": mse_test.item(),
            "training_time": training_time,
            "inference_time": inference_time
        })

        val_table = [
            [outer_fold, val_iid, seed, saved_epoch, saved_lr, n_hid, n_layers, patience, factor, dropout,
            l1_train.item(), mse_train.item(), l1_val.item(), mse_val.item(), l1_test.item(), mse_test.item()]
        ]
        val_columns = [
            'outer_fold', 'val_iid', 'seed', 'saved_epoch', 'saved_lr', 'n_hid', 'n_layers', 
            'patience', 'factor', 'dropout', 'l1_train', 'mse_train', 'l1_val', 'mse_val', 
            'l1_test', 'mse_test'
        ]
        val_table_df = pd.DataFrame(val_table, columns=val_columns)
        val_file_path = f'{out_loc}/runtime/val_test_table_seed_{seed}.csv'

        if not os.path.exists(val_file_path):
            val_table_df.to_csv(val_file_path, index=False, mode='w', header=True)
        else:
            val_table_df.to_csv(val_file_path, index=False, mode='a', header=False)


    predictions_save_path = f'{out_loc}/predictions/{seed}_{n_hid}_{n_layers}_{dropout}.pkl'
    with open(predictions_save_path, 'wb') as f:
        torch.save(all_predictions, f)
    

    avg_l1_test = np.mean([p['l1_test'] for p in all_predictions])
    avg_mse_test = np.mean([p['mse_test'] for p in all_predictions])
    std_l1_test = np.std([p['l1_test'] for p in all_predictions])
    std_mse_test = np.std([p['mse_test'] for p in all_predictions])
    
    summary_table = [[seed, n_hid, n_layers, dropout, avg_l1_test, std_l1_test, avg_mse_test, std_mse_test]]
    summary_columns = ['seed', 'n_hid', 'n_layers', 'dropout', 'avg_l1_test', 'std_l1_test', 'avg_mse_test', 'std_mse_test']
    summary_df = pd.DataFrame(summary_table, columns=summary_columns)
    summary_path = f'{out_loc}/runtime/summary_seed_{seed}.csv'
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("example: python main.py 42 64 2 0.2")
        sys.exit(1)

    seed = int(sys.argv[1])
    n_hid = int(sys.argv[2])
    n_layers = int(sys.argv[3])
    dropout = float(sys.argv[4])

    train_predict(seed, n_hid, n_layers, dropout)