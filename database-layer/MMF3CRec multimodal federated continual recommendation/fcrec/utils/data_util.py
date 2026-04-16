import numpy as np
import pickle
import torch
import os
import pandas

def load_pickle(file_path):
    with open(file_path,"rb") as f:
        return pickle.load(f)
    
    
def save_pickle(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    
    
def load_data_as_dict(data_dict_path, num_task = 6):
    
    total_train_dataset = dict()
    total_valid_dataset = dict()
    total_test_dataset = dict()
    total_item_list = dict()
    
    for task_idx in range(num_task):
        task_data_dict_path = os.path.join(data_dict_path, f"TASK_{task_idx}.pickle")
        task_data = load_pickle(task_data_dict_path)
        total_train_dataset[f"TASK_{task_idx}"] = task_data["train_dict"]
        total_valid_dataset[f"TASK_{task_idx}"] = task_data["valid_dict"]
        total_test_dataset[f"TASK_{task_idx}"] = task_data["test_dict"]
        total_item_list[f"TASK_{task_idx}"] = task_data["item_list"]
        
    
    return total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list