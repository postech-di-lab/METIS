import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Run Recommender")

    parser.add_argument('--recommender', nargs='?', help='Choose a recommender.', required=True)
    parser.add_argument('--dataset', nargs='?', help='Choose a dataset.', required=True)        
    parser.add_argument('--lRate', type=float, default=0.001, help='Learning rate.', required=True)
    parser.add_argument('--mode', nargs='?', help='Validation or Test (Val, Test)', required=True)
    parser.add_argument('--early_stop', type=int, default=50, help='Early stop iteration.')
    parser.add_argument('--topK', nargs='?', default='[1,5,10,20,50]', help="topK")
    parser.add_argument('--numEpoch', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative samples.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size.')
    parser.add_argument('--batchSize_test', type=int, default=2000, help='Batch size for test.')
    parser.add_argument('--cuda', type=int, default=0, help='Speficy GPU number')
    parser.add_argument('--reg1', type=float, default=0.01, help='Distance Regularization.')
    parser.add_argument('--reg2', type=float, default=0.01, help='Neighborhood Regularization.')
    parser.add_argument('--embedding_dim', type=int, default=10, help='Number of embedding dimensions.')
    parser.add_argument('--rand_seed', type=int, default=34567, help='Random seed.')
    
    
    
    return parser.parse_known_args()

def printConfig(args):
    common_elems = ['recommender', 'dataset', 'numEpoch', 'lRate', 'num_negatives', 'embedding_dim', 'early_stop', 'batch_size', 'reg1', 'reg2', 'rand_seed', 'margin']
    
    rec = args.recommender
    
    st = []
    for elem in common_elems:
        s = str(elem + ": " + str(getattr(args, elem)))
        st.append(s)
    print(st)
        
if __name__ == '__main__':
    args, unknown = parse_args()
    
    printConfig(args)
    
    if args.recommender == 'TransCF':
        from TransCF import TransCF
        recommender = TransCF(args)
        
    recommender.training()
