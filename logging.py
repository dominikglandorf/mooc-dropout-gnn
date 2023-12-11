import csv
import os
from datetime import datetime

def append_to_csv(filename, dataset_path, relative_time, semi_transductive, heterogenous_msg_passing,
                  batch_size, num_epochs, learning_rate, emb_dim, hidden_dim, time_dim, identity_dim,
                  train_loss, val_loss, test_loss, train_AUC, val_AUC, test_AUC, train_ap, val_ap, test_ap, run):
    if not os.path.exists(filename):
        print("creat_file")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['timestamp', "dataset_path", 'relative_time', "semi_transductive", "heterogenous_msg_passing",
                      "batch_size", "num_epochs", "learning_rate", "emb_dim", "hidden_dim", "time_dim", "identity_dim",
                      'train_loss', 'val_loss', 'test_loss', 'train_AUC', 'val_AUC', 'test_AUC',
                      'train_AP', 'val_AP', 'test_AP', 'run']
            writer.writerow(header)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = [timestamp, dataset_path, relative_time, semi_transductive, heterogenous_msg_passing,
                batch_size, num_epochs, learning_rate, emb_dim, hidden_dim, time_dim, identity_dim,
                train_loss, val_loss, test_loss, train_AUC, val_AUC, test_AUC, train_ap, val_ap, test_ap,run]
        writer.writerow(data)