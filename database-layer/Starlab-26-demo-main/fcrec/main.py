import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.util import *
from utils.data_util import *
from utils.data_loader import *

import argparse

# ===================================================================== FedMF =====================================================================
from baseline.fedmf.fedmf import *

# ===================================================================== FedMLP =====================================================================
from baseline.fedmlp.fedmlp import *

# ===================================================================== PFedRec =====================================================================
from baseline.pfedrec.pfedrec import *


def fcrec(args):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root=os.path.dirname(current_file_dir)

    args.num_task = 4 
    
    device = get_device(args, threshold=500)


    data_block_path = os.path.join(project_root, "fcrec_data", "dataset", args.dataset, "total_blocks_timestamp.pickle")
    data_dict_path = os.path.join(project_root, "fcrec_data", "dataset", args.dataset)

    
    total_blocks = load_pickle(data_block_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)

    num_user_item_info = get_num_user_item(total_blocks)

    args.num_user, args.num_item = num_user_item_info['TASK_0']['num_user'], num_user_item_info['TASK_0']['num_item']
    
    is_shuffle = False

    args.layers = [args.dim * 2, args.dim // 2]

    if args.mode == 'vision':
        vision_embs = torch.load(args.vision_embedding_path)
        vision_embs = F.normalize(vision_embs, p=2, dim=1)
        embs = vision_embs
    elif args.mode == 'text':
        text_embs = torch.load(args.text_embedding_path)
        text_embs = F.normalize(text_embs, p=2, dim=1)
        embs = text_embs
    elif args.mode == 'concat':
        vision_embs = torch.load(args.vision_embedding_path)
        text_embs = torch.load(args.text_embedding_path)
        embs = torch.cat([vision_embs, text_embs], dim=1)
        embs = F.normalize(embs, p=2, dim=1)
    else:
        vision_embs = torch.load(args.vision_embedding_path)
        text_embs = torch.load(args.text_embedding_path)
        embs = vision_embs + text_embs
        embs = F.normalize(embs, p=2, dim=1)
    
    # ===================================================================== FedMF =====================================================================    
    if args.baseline == 'fedmf_fcrec': 
        engine = FedMF_Engine(args, device, embs)
    
    # ===================================================================== FedNCF =====================================================================    
    elif args.baseline == 'fedmlp_fcrec':
        engine = FedMLP_Engine(args, device)
        
    # ===================================================================== PFedRec =====================================================================    
    elif args.baseline == 'pfedrec_fcrec':
        engine = PFedRec_Engine(args, device)
    
    
    input_total_data = [total_train_dataset, total_valid_dataset, total_test_dataset]
    
    if args.load_model == 0 and args.save_model == 1:
        engine.run(task=0, input_total_data=input_total_data, is_base = True)
        save_baseblock_param(args, engine)
        print("Save complete")
        exit()
    
    elif args.load_model == 1 and args.save_model == 0:
        engine = load_baseblock_param(args)
        engine.args = args
        
        for task in range(1, args.num_task):
            is_base = False
        
            args.num_user, args.num_item = num_user_item_info[f'TASK_{task}']['num_user'], num_user_item_info[f'TASK_{task}']['num_item']
            engine.add_user_item(args.num_user, args.num_item)
            
            engine.run(task, input_total_data, is_base)
        
    
    elif args.load_model == 0 and args.save_model == 0: 
        for task in range(args.num_task):   
            is_base = True if task == 0 else False    
            
            if not is_base:
                args.num_user, args.num_item = num_user_item_info[f'TASK_{task}']['num_user'], num_user_item_info[f'TASK_{task}']['num_item']
                engine.add_user_item(args.num_user, args.num_item)
            
            engine.run(task, input_total_data, is_base)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_valid', action='store_true', default=False)
    parser.add_argument('--save_model', type = int, default=0, help='0: not save, 1: save')
    parser.add_argument('--load_model', type = int, default=1, help='0: scratch, 1: load')
    parser.add_argument('--save_result', type = int, default=0, help='0: not save, 1: save')
    
    parser.add_argument('--dataset', '--dd', type = str, default = 'ml-100k', choices = ['ml-100k', 'hetrec2011', 'lastfm-2k','ml-latest-small'])
    parser.add_argument('--optimizer', '--o', type = str, default = 'SGD') 
    parser.add_argument('--dim', '--d', type=int, default = 32)
    
    parser.add_argument('--clients_sample_ratio', '--c', type=int, default=1.0, help='# participants in a round')
    parser.add_argument('--batch_size', '--b', type=int, default=512, help='# item in a batch')
    parser.add_argument('--num_round', '--r', type=int, default=100, help='ml-100k, HetRec211: 100, ML-Latest-Small, lastfm-2k: 150')
    parser.add_argument('--local_epoch', '--e', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--num_ns', '--nn', type=int, default=4)   
    parser.add_argument('--standard', type=str, default="NDCG")
    parser.add_argument('--lr', type=float, default=0.5) 
    parser.add_argument('--lr_eta', type=float, default=170)
    parser.add_argument('--weight_decay', type=float, default=0)
    # ----------------------------------------Fed parameter-----------------------------------------------------------> 
    parser.add_argument('--dp', type=float, default = 0.0)
    # ----------------------------------------model specific parameter-----------------------------------------------------------> 
    parser.add_argument('--reg_mlp', action='store_true', default=True) 
    # -------------------------------------------KD related parameter-------------------------------------------------------->     
    parser.add_argument('--client_cl', action='store_true', default=True)
    parser.add_argument('--server_cl', action='store_true', default=True)
    parser.add_argument('--beta',type=float, default=0.2)
    parser.add_argument('--eps', type=float, default=1e-3, help = 'epsilon for ranking discrepancy rate')
    parser.add_argument('--topN', type=int, default=30)
    parser.add_argument('--reg_client_cl', type=float, default=1e-3)
    # -----------------------------------------------------------
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--specific_gpu', type=int, default=1, help='0: for scheduler, 1: assign specific gpu')
    parser.add_argument('--tqdm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone', type=str, default='fedmf', choices=['fedmf', 'fedmlp', 'pfedrec'])
    parser.add_argument('--model', type=str, default='fcrec')
    parser.add_argument('--reg_reduction', type=str, default='sum', choices = ['sum', 'mean'])
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--job_id', type=str, default='000')
    # -----------------------------------------------------------
    parser.add_argument('--mode', type=str, default='both', choices=['vision','text','both','concat'])
    parser.add_argument('--clip_dim', type=int, default=512, help='Dimension of pre-extracted CLIP embeddings')
    parser.add_argument('--alpha', type=float, default=0.3, help='Fixed weight for visual embedding')  
    parser.add_argument('--proj_dim', type=int, default=256, help='Dimension of the projection layer for visual features in FedMF') 
    parser.add_argument('--proj_lr_ratio', type=float, default=0.01)
    parser.add_argument('--prefix', type=str, default='', help='prefix for the saved model file')
    parser.add_argument('--vision_embedding_path', type=str, default='./embeddings/movie_image_embeddings_512.pt')
    parser.add_argument('--text_embedding_path', type=str, default='./embeddings/movie_text_embeddings_512.pt')


    args = parser.parse_args()
    
    
    
    args.baseline = args.backbone + '_' + args.model
        
    if args.save_model == 1 and args.load_model == 1:
        
        exit()

    set_seed(args.seed)
    
    start = time.time()
    print(f"seed: {args.seed}")
    print(args)
    fcrec(args)
    
    end = time.time()
    
