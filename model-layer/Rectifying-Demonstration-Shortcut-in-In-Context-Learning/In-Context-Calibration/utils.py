import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num, seed):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    # if num > len(labels):
    #     assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    if num > len(labels):
        print(f"Warning: you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}. Setting num to {len(labels)} instead.")
        num = len(labels)
    np.random.seed(seed=seed)
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

'''
Sampling Function for Chaining Experiments
'''

def random_sampling_by_label(sentences, labels, num, seed, params):
    """randomly sample subset of the training pairs for a randomly chosen label"""
    assert 'label_dict' in params, "params should contain a 'label_dict' key"
    
    # Randomly select one unique label from params['label_dict']
    unique_labels = list(params['label_dict'].keys())
    np.random.seed(seed=seed)
    chosen_label = np.random.choice(unique_labels)
    
    # Extract sentences corresponding to the chosen label
    label_specific_sentences = deepcopy([sent for idx, sent in enumerate(sentences) if labels[idx] == chosen_label])
    label_specific_labels = deepcopy([chosen_label for _ in range(len(label_specific_sentences))])

    sampled_sentences, sampled_labels = random_sampling(label_specific_sentences, label_specific_labels, num, seed)
    return sampled_sentences, sampled_labels, chosen_label  # Also return the chosen label

from rank_bm25 import BM25Okapi

def whitespace_tokenizer(text):
    return text.lower().split() 

def bm25_rank(test_sentences, train_sentences):
    """Rank test sentences based on BM25 score relative to train_sentences and return the sorted indices."""
    bm25 = BM25Okapi(train_sentences, tokenizer=whitespace_tokenizer)
    combined_scores = [0] * len(test_sentences)
    
    # Compute the BM25 score for every sentence in the test_sentences using the train_sentences as reference
    for test_sentence in test_sentences:
        scores = bm25.get_scores(whitespace_tokenizer(test_sentence))
        combined_scores[test_sentences.index(test_sentence)] = sum(scores)

    # Sort the indices of the test_sentences in accordance with their BM25 scores
    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    
    assert len(sorted_indices) == len(test_sentences), "Length mismatch between sorted indices and test sentences"
    
    return sorted_indices

def random_sampling_exclude_label(sentences, labels, num, excluded_label, params, train_sentences):
    """randomly sample hardest negative subset of the test sentences excluding a specific label"""
    
    # Extract sentences NOT corresponding to the excluded label
    non_excluded_sentences = [sent for idx, sent in enumerate(sentences) if labels[idx] != excluded_label]
    non_excluded_labels = [lbl for idx, lbl in enumerate(labels) if lbl != excluded_label]

    sampling_upper_limits = int(0.3*len(non_excluded_labels))

    if num > sampling_upper_limits:
        print(f"Warning: you tried to randomly sample {num}, which is more than the total size of the pool {sampling_upper_limits}. Setting num to {sampling_upper_limits} instead.")
        num = sampling_upper_limits
    
    # Use BM25 to rank the sentences based on train sentences
    sorted_indices = bm25_rank(non_excluded_sentences, train_sentences)
    
    # Get the hardest negatives based on the provided `num` value
    hardest_negative_sentences = [non_excluded_sentences[i] for i in sorted_indices[:num]]
    hardest_negative_labels = [non_excluded_labels[i] for i in sorted_indices[:num]]

    assert excluded_label not in hardest_negative_labels, f"{excluded_label} is in hard negatives"

    return hardest_negative_sentences, hardest_negative_labels


def stratify_random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    
    # Determine the number of classes
    n_classes = len(np.unique(labels))
    
    # Check if the requested sample size is a multiple of the number of classes
    if num % n_classes != 0:
        print(f"Warning: the number to sample {num} is not a multiple of the number of classes {n_classes}.")
        print(f"Re assigning sampling {num} to a multiple of the number of classes {n_classes}.(i.e. *2)")
        num = n_classes * 2

    # Determine the number of samples per class
    samples_per_class = num // n_classes

    # Perform stratified sampling
    selected_sentences = []
    selected_labels = []
    for label in np.unique(labels):
        idxs = np.where(np.array(labels) == label)[0]
        if len(idxs) < samples_per_class:
            print(f"Warning: not enough samples in class {label} for stratified sampling.")
            return None, None
        selected_idxs = np.random.choice(idxs, size=samples_per_class, replace=False)
        selected_sentences.extend([sentences[i] for i in selected_idxs])
        selected_labels.extend([labels[i] for i in selected_idxs])
    
    return selected_sentences, selected_labels  #[joonwon] return 받은 sentence shuffle해서 사용.

langauge_model = None
language_model_tokenizer = None
def setup_gpt2(model_name):
    # load the GPT-2 model
    global langauge_model
    global language_model_tokenizer
    if langauge_model is None:
        print("Setting up language model")
        assert model_name in ["EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6b"]
        # size = model_name.lower.split('-')
        # size에서 숫자 걸러내야 하지 않음,,,? 
        if model_name in ["EleutherAI/gpt-neo-2.7B"]:
            langauge_model = AutoModelForCausalLM.from_pretrained(model_name)
            langauge_model.eval().to(device)
        elif model_name in ["EleutherAI/gpt-j-6b"]:
            checkpoint = model_name
            config = AutoConfig.from_pretrained(checkpoint)
            with init_empty_weights():
                langauge_model = AutoModelForCausalLM.from_config(config)
            langauge_model.tie_weights()
            langauge_model = load_checkpoint_and_dispatch(
            langauge_model, "sharded-"+ model_name.split('/')[-1], device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        
        language_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        language_model_tokenizer.padding_side = "left"
        language_model_tokenizer.pad_token = language_model_tokenizer.eos_token
        langauge_model.config.pad_token_id = langauge_model.config.eos_token_id
        print("Finished")

def setup_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key

def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = language_model_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = langauge_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False) # repetition_penalty=1.85, no_repeat_ngram_size=3
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != language_model_tokenizer.pad_token_id).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the context and the next l tokens
        logits = langauge_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu() # You should task care for positional embedding for other models.
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = language_model_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = language_model_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(language_model_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[language_model_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[language_model_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(language_model_tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs 
                    curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

        choices.append(curr_json)
    return_json['choices'] = choices
    return return_json

def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='\n', n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
    return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt3' not  in model:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        setup_gpt2(model)
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    else:
        setup_gpt3()
        return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def construct_prompt(params, train_sentences, train_labels, test_sentence, prefix=True):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    if not prefix:
        prompt = ''
        q_prefix = ''
        a_prefix = ' '
    else:
        prompt = params["prompt_prefix"] # instruction
        q_prefix = params["q_prefix"]
        a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def construct_prompt_with_domain(params, train_sentences, train_labels, test_sentence, domain, prefix=True):

    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    if not prefix:
        prompt = ''
        q_prefix = ''
        a_prefix = ' '
    else:
        prompt = params["prompt_prefix"] # instruction
        q_prefix = params["q_prefix"]
        a_prefix = params["a_prefix"]
    position = np.random.randint(0,len(train_sentences))
    for index, (s, l) in enumerate(zip(train_sentences, train_labels)):
        if index == position:
            prompt += domain
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    else:
        prompts = override_prompt

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original F1  ','Context Calibrated F1', 'Domain Calibrated (Demonstration) F1', 'Domain Calibrated (Test set) F1', 'In-Context Calibrated F1')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                num_ratio_node = num_shots_node[num_shots]
                for ratio in num_ratio_node.keys():
                    f1_score = np.array(list(num_ratio_node[ratio].values()))
                    f1_score_mean = np.mean(f1_score, axis=0)
                    f1_score_low = np.min(f1_score, axis=0)
                    f1_score_high = np.max(f1_score, axis=0)
                    f1_score_std = np.std(f1_score, axis=0)

                    print(f"\n{num_shots}-shot, {ratio}-%, {len(f1_score)} seeds")
                    for i, (m, l, h, s) in enumerate(zip(f1_score_mean, f1_score_low, f1_score_high, f1_score_std)):
                        print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")

                    print()

# def print_results(tree, names=('Original Accuracy  ','Context Calibrated Accuracy', 'Domain Calibrated Accuracy')):
#     # print out all results
#     root = deepcopy(tree)
#     for dataset in root.keys():
#         print(f"\n\nDataset: {dataset}")
#         models_node = root[dataset]
#         for model in models_node.keys():
#             print(f"\nModel: {model}")
#             num_shots_node = models_node[model]
#             for num_shots in num_shots_node.keys():
#                 accuracies = np.array(list(num_shots_node[num_shots].values()))
#                 accuracies_mean = np.mean(accuracies, axis=0)
#                 accuracies_low = np.min(accuracies, axis=0)
#                 accuracies_high = np.max(accuracies, axis=0)
#                 accuracies_std = np.std(accuracies, axis=0)

#                 print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
#                 for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
#                     print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")

#                 print()


def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['f1_scores']
    print_results(result_tree)

class PrintLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Flush the terminal output
        with open(self.log_file, 'a') as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()