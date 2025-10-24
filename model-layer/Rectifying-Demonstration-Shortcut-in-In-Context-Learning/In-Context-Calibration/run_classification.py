'''
Our Implementation is largely based on the implementation of the 
''https://github.com/tonyzhaozh/few-shot-learning''.

Additionally, we have integrated Domain Calibration into our framework. 
Supplementary files, including data_utils.py and README.txt, are currently being refined and will soon be made available on a public GitHub repository for further exploration.

We also suggest using various model loading configurations (e.g., float16) to accommodate different GPU environments.
'''

import argparse
from typing import List
from data_utils import load_dataset
from utils import *

def main(models, datasets, all_shots, num_seeds, subsample_test_set, lambda1, api_num_log_prob, approx, do_task_learning, use_saved_results, bs): 
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs,
        'do_task_learning': do_task_learning,
    }

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                for seed in range(num_seeds):       
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['num_shots'] = num_shots
                    p['lambda1'] = lambda1
                    p['seed'] = seed
                    p['expr_name'] = f"{p['dataset']}_{get_model_name(p['model'])}_{p['num_shots']}shot_{p['lambda1']}lambda_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}" 
                    all_params.append(p)

    if use_saved_results:
        load_results(all_params)
    else:
        log_dir = f"./log_{get_model_name(p['model'])}-{p['lambda1']}-{do_task_learning}"
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_file = f"./log_{get_model_name(p['model'])}-{p['lambda1']}-{do_task_learning}/{p['dataset']}"
        logger = PrintLogger(log_file)
        sys.stdout = logger
        save_results(all_params)
        
def get_model_name(model_path):
    if "/" in model_path:
        return model_path.split("/")[-1]
    else:
        return model_path

def save_results(params_list, freeze_test_set=False):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    # query the model and save the responses
    result_tree = dict()
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])

        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)

        ### sample few-shot training examples
        np.random.seed(params['seed'])
        print(f"Seed - {params['seed']}")
        train_sentences, train_labels = random_sampling(sentences=all_train_sentences, labels=all_train_labels, num=params['num_shots'], seed=params['seed'])
        
        ### sample test set
        if params['subsample_test_set'] is None:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0) # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
                test_sentences, test_labels = random_sampling(sentences=all_test_sentences, labels=all_test_labels, num=params['subsample_test_set'], seed=params['seed'])
                print(f"selecting {len(test_labels)} subsample of test set")

        ### if do_task_learning, modifiy label_dict and inv_label_dict
        if params['do_task_learning']:
            params['label_dict'], params['inv_label_dict'] = generate_task_learning_dicts(n=len(params['label_dict']))

        # Sanity Check before Experiments
        params_check(params)

        ### Evaluate the performance and save all results
        print(f"getting raw resp for {len(test_sentences)} test sentences")
        
        raw_resp_test = get_model_response(params, train_sentences, train_labels, test_sentences)

        # get prob for each label
        all_label_probs = get_label_probs(params, raw_resp_test, train_sentences, train_labels, test_sentences)

        ### Contextual Calibration
        print(f"contextual calibration for {len(test_sentences)} test sentences")
        content_free_inputs = ["N/A"]

        # calculate P_cf
        p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)

        # Domain Calibration (tr)
        print(f"Domain calibration (tr) for {len(test_sentences)} test sentences") 
        p_df_tr = get_domain_probs_demonstration(params, train_sentences, train_labels, test_sentences, 20) 

        # Domain Calibration (te)
        print(f"Domain calibration (te) for {len(test_sentences)} test sentences") 
        p_df_te = get_domain_probs_original(params, train_sentences, train_labels, test_sentences, 20) 

        # In-Context Calibration
        print(f"In-Context Calibration for {len(test_sentences)} test sentences")
        p_icc = get_calibration_probs_in_context(params, train_sentences, train_labels)

        acc_original, f1_original = eval_accuracy(all_label_probs, test_labels)
        acc_calibrated,f1_context = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf)
        acc_dc_tr, f1_domain_tr = eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None, p_df=p_df_tr) 
        acc_dc_te, f1_domain_te = eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None, p_df=p_df_te) 
        acc_icc, f1_icc = eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None, p_df=p_icc)

        accuracies = [acc_original, acc_calibrated, acc_dc_tr, acc_dc_te, acc_icc]
        f1_scores =[f1_original, f1_context, f1_domain_tr, f1_domain_te, f1_icc]
        print(f"Accuracies: {accuracies}")
        print(f"Macro-F1: {f1_scores}")
        print(f"Average Label Estimation Before Calibration: {np.mean(np.array(deepcopy(all_label_probs)),axis=0)}")
        print(f"p_cf      : {p_cf}")
        print(f"p_df_tr      : {p_df_tr}")
        print(f"p_df_te     : {p_df_te}")
        print(f"p_icc       : {p_icc}")

        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots'], params['lambda1']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        # node[params['seed']] = accuracies
        node[params['seed']] = f1_scores

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['train_sentences'] = train_sentences
        result_to_save['train_labels'] = train_labels
        result_to_save['test_sentences'] = test_sentences
        result_to_save['test_labels'] = test_labels
        result_to_save['raw_resp_test'] = raw_resp_test
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['p_cf'] = p_cf
        result_to_save['p_df_tr'] = p_df_tr
        result_to_save['p_df_te'] = p_df_te
        result_to_save['p_icc'] = p_icc
        result_to_save['accuracies'] = accuracies
        result_to_save['f1_scores'] = f1_scores
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        save_pickle(params, result_to_save)

    print_results(result_tree)

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None, p_df=None):
    from sklearn.metrics import f1_score
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None or p_df is not None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    elif p_cf is not None:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
    if p_df is not None:
        assert all_label_probs.shape[1] == len(p_df)
        all_label_probs = all_label_probs/p_df

    prediciton_list = []
    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        prediciton_list.append(ans_label)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    assert len(prediciton_list) == len(test_labels)            
    macro_f1 = f1_score(test_labels, prediciton_list, average='macro')
    return np.mean(correctness_list), macro_f1

def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # No Indent for llama
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label # No Indent for llama
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"
    
    row_sums = np.sum(all_label_probs, axis=1)
    row_sums = row_sums[:, np.newaxis]
    all_label_probs = all_label_probs / row_sums
    return all_label_probs # NOT NORMALIZED

def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)

        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1]) # No Indent for llama
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y

def get_domain_probs_demonstration(params, train_sentences, train_labels, test_sentences, num_of_calibration=20):
    """Query model with domain free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_sentences).toarray() # in-domain-sampling from train_sentences
    feature_names = vectorizer.get_feature_names_out()
    average_word = int(np.round(np.average(np.sum(X, axis=-1))))
    # Get the frequency of each word
    word_frequencies = np.asarray(X.sum(axis=0)).ravel()
    # Normalize the frequencies to get probabilities
    word_probabilities = word_frequencies / word_frequencies.sum()
    in_domain_inputs = [" ".join(np.random.choice(feature_names, size=average_word, 
                                                  replace=True, p=word_probabilities)) for i in range(num_of_calibration)]
    print("original in domain input: ", in_domain_inputs)

    all_p_y = []
    for domain_free_input in in_domain_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, domain_free_input)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1]) # No Indent for llama
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y

def get_domain_probs_original(params, train_sentences, train_labels, test_sentences, num_of_calibration=20):
    """Query model with domain free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(test_sentences).toarray() # in-domain-sampling from test_sentences
    feature_names = vectorizer.get_feature_names_out()
    average_word = int(np.round(np.average(np.sum(X, axis=-1))))
    # Get the frequency of each word
    word_frequencies = np.asarray(X.sum(axis=0)).ravel()
    # Normalize the frequencies to get probabilities
    word_probabilities = word_frequencies / word_frequencies.sum()
    in_domain_inputs = [" ".join(np.random.choice(feature_names, size=average_word, 
                                                  replace=True, p=word_probabilities)) for i in range(num_of_calibration)]
    print("original in domain input: ", in_domain_inputs)

    all_p_y = []
    for domain_free_input in in_domain_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, domain_free_input)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1]) # No Indent for llama
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y

def get_calibration_probs_in_context(params, train_sentences, train_labels): 
    label_dict = params['label_dict']
    lambda1 = float(params['lambda1'])
    num_shots = len(train_sentences)

    all_replaced_sentences = []
    all_replaced_labels = []    
    
    for i in range(num_shots):

        # Extract all sentences and labels except the k-th one
        temp_sentences_before, temp_sentences_after = train_sentences[:i], train_sentences[i+1:]
        temp_labels_before, temp_labels_after = train_labels[:i], train_labels[i+1:]

        x_k, y_k = train_sentences[i], train_labels[i]
        replaced_sentences_list = []
        replaced_labels_list = []

        replaced_sentences_list.extend(temp_sentences_before)
        replaced_sentences_list.extend(temp_sentences_after)
        replaced_labels_list.extend(temp_labels_before)
        replaced_labels_list.extend(temp_labels_after)

        all_replaced_sentences.append({x_k:replaced_sentences_list})
        all_replaced_labels.append({y_k:replaced_labels_list})


    all_p_y = []
    for x, y in zip(all_replaced_sentences, all_replaced_labels):
        x_k = list(x.keys())[0] # k-th input
        y_k = list(y.keys())[0] # k-th label
        replaced_demonstration = x[x_k]
        replaced_label_dist = y[y_k]
        all_k_1_p_y = []
        for i in range(2):
            p_y = [0] * len(label_dict)
            if i == 0:
                pass
            else:
                x_k = x_k.split()
                np.random.shuffle(x_k)
                x_k = ' '.join(x_k)
            prompt = construct_prompt(params, replaced_demonstration, replaced_label_dist, x_k)
            for i, answers in label_dict.items():
                prob = 0
                for a in answers:
                    prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
                p_y[i] = prob
            all_k_1_p_y.append(p_y)
        all_k_1_p_y = [np.array(p, dtype=float) for p in all_k_1_p_y]
        k_1_p_y  = lambda1 * all_k_1_p_y[0] + (1 - lambda1) * all_k_1_p_y[1] 
        all_p_y.append(k_1_p_y)
    
    p_lm = np.mean(np.array(all_p_y), axis=0)
    p_lm = p_lm / np.sum(p_lm) # normalize
    return p_lm

def params_check(params):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, params['model'], echo=True, num_log_probs=2)['choices'][0]['logprobs']['tokens'][0]
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

    if not (params['dataset'] in ['cb', 'rte', 'wnli', 'anli', 'sick']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

def average_token_length(test_sentences: List[str], language_model_tokenizer) -> int:
    avg_list = list()

    for sentence in test_sentences:
        tokens = language_model_tokenizer.tokenize(sentence)
        avg_list.append(len(tokens))

    return int(np.mean(avg_list))

def generate_task_learning_dicts(n, symbols=None):
    # symbols will be ['A', 'B', 'C'] or ['0', '1', '2']
    if symbols is None:
        symbols = [str(i) for i in range(0, n)]
        np.random.shuffle(symbols) # Let each label mapped to random symbol

    if len(symbols) < n:
        raise ValueError("Not enough symbols for the desired length.")

    label_dict = {}
    inv_label_dict = {}

    for i in range(n):
        label_dict[i] = [symbols[i]]
        inv_label_dict[symbols[i]] = i

    return label_dict, inv_label_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument('--lambda1', dest='lambda1', action='store', required=True, help='lambda > this is for ablation study.')

    # For shuffling ablation
    # parser.add_argument('--shuffle', dest='shuffle', action='store', required=True, help='shuffle> this is for ablation study.')
    parser.add_argument('--do_task_learning', dest='do_task_learning', action='store_const', const=True, default=False,
                        help='whether to perform task learning')
    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]
    
    # simple processing
    def convert_to_floatlist(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [float(s.strip()) for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)