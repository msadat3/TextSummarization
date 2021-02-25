from Utils import *
import os.path as p
import os
import pandas
import numpy as np
from transformers import PegasusTokenizer, BartTokenizer

base = '/home/ubuntu/CNNDM/'
traininingSet_location = base+'train.csv'
test_set_location = base+'test.csv'
validation_set_location = base + 'validation.csv'

traininingSet = pandas.read_csv(traininingSet_location)
testingSet = pandas.read_csv(test_set_location)
validationSet = pandas.read_csv(validation_set_location)

model_type = 'BART'
prertained_model_name = 'facebook/bart-base'
if model_type == 'PEGASUS':
    tokenizer = PegasusTokenizer.from_pretrained(prertained_model_name, do_lower_case=True)
elif model_type == 'BART':
    tokenizer = BartTokenizer.from_pretrained(prertained_model_name, do_lower_case=True)

def Tokenize_Input(text):
    #print(len(text))
    #first_encoded = tokenizer.encode(first_sentence,add_special_tokens=False)
    text_encoded = tokenizer.encode(text, truncation=True, padding=True, add_special_tokens=True)
    return text_encoded

def get_attention_masks(X,sourceOrSummary):
    attention_masks = []

    # For each sentence...
    for sent in X:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id != tokenizer.pad_token_id) for token_id in sent]
        #if sourceOrSummary =='summary':
           # att_mask.insert(0, 0)#(index,value)
        # Store the attention mask for this sentence.
        att_mask = np.asarray(att_mask)
        attention_masks.append(att_mask)
    attention_masks = np.asarray(attention_masks)
    return attention_masks

def pad_seq(seq,max_len,pad_idx):
    if len(seq)>max_len:
        end_idx = seq[-1]
        seq = seq[0:max_len-1]
        seq.append(end_idx)
    while len(seq) != max_len:
        seq.append(pad_idx)
    return seq

def prepare_all_data(output_location):
    if p.exists(output_location) == False:
        os.mkdir(output_location)

    X_train_source = traininingSet.apply(lambda x: Tokenize_Input(x['source']), axis=1)
    X_train_summary = traininingSet.apply(lambda x: Tokenize_Input(x['summary']), axis=1)

    X_test_source = testingSet.apply(lambda x: Tokenize_Input(x['source']), axis=1)
    X_test_summary = testingSet.apply(lambda x: Tokenize_Input(x['summary']), axis=1)

    X_valid_source = validationSet.apply(lambda x: Tokenize_Input(x['source']), axis=1)
    X_valid_summary = validationSet.apply(lambda x: Tokenize_Input(x['summary']), axis=1)

    X_train_source = pandas.Series(X_train_source)
    X_train_summary = pandas.Series(X_train_summary)

    X_test_source = pandas.Series(X_test_source)
    X_test_summary = pandas.Series(X_test_summary)

    X_valid_source = pandas.Series(X_valid_source)
    X_valid_summary = pandas.Series(X_valid_summary)

    max_len_source = 0
    max_len_summary = 0
    for x in X_train_source:
        if len(x) > max_len_source:
            max_len_source = len(x)
    for x in X_train_summary:
        if len(x) > max_len_summary:
            max_len_summary = len(x)

    print('max', max_len_source, max_len_summary)

    X_train_source = X_train_source.apply(pad_seq, max_len=max_len_source, pad_idx=tokenizer.pad_token_id)
    X_train_summary = X_train_summary.apply(pad_seq, max_len=max_len_summary, pad_idx=tokenizer.pad_token_id)

    X_test_source = X_test_source.apply(pad_seq, max_len=max_len_source, pad_idx=tokenizer.pad_token_id)
    X_test_summary = X_test_summary.apply(pad_seq, max_len=max_len_summary, pad_idx=tokenizer.pad_token_id)

    X_valid_source = X_valid_source.apply(pad_seq, max_len=max_len_source, pad_idx=tokenizer.pad_token_id)
    X_valid_summary = X_valid_summary.apply(pad_seq, max_len=max_len_summary, pad_idx=tokenizer.pad_token_id)

    X_train_source = np.array(X_train_source.values.tolist())
    X_train_summary = np.array(X_train_summary.values.tolist())

    X_test_source = np.array(X_test_source.values.tolist())
    X_test_summary = np.array(X_test_summary.values.tolist())

    X_valid_source = np.array(X_valid_source.values.tolist())
    X_valid_summary = np.array(X_valid_summary.values.tolist())

    att_mask_train_source= get_attention_masks(X_train_source,'source')
    att_mask_train_summary = get_attention_masks(X_train_summary, 'summary')

    att_mask_test_source = get_attention_masks(X_test_source,'source')
    att_mask_test_summary = get_attention_masks(X_test_summary, 'summary')

    att_mask_valid_source = get_attention_masks(X_valid_source,'source')
    att_mask_valid_summary = get_attention_masks(X_valid_summary, 'summary')

    save_data(X_train_source, output_location + 'X_train_source.pkl')
    save_data(X_train_summary, output_location + 'X_train_summary.pkl')

    save_data(X_test_source, output_location + 'X_test_source.pkl')
    save_data(X_test_summary, output_location + 'X_test_summary.pkl')

    save_data(X_valid_source, output_location + 'X_valid_source.pkl')
    save_data(X_valid_summary, output_location + 'X_valid_summary.pkl')



    save_data(att_mask_train_source, output_location + 'att_mask_train_source.pkl')
    save_data(att_mask_train_summary, output_location + 'att_mask_train_summary.pkl')

    save_data(att_mask_test_source, output_location + 'att_mask_test_source.pkl')
    save_data(att_mask_test_summary, output_location + 'att_mask_test_summary.pkl')

    save_data(att_mask_valid_source, output_location + 'att_mask_valid_source.pkl')
    save_data(att_mask_valid_summary, output_location + 'att_mask_valid_summary.pkl')

    print(X_train_source.shape, att_mask_train_source.shape)
    print(X_train_summary.shape, att_mask_train_summary.shape)

    print(X_test_source.shape, att_mask_test_source.shape)
    print(X_test_summary.shape, att_mask_test_summary.shape)

    print(X_valid_source.shape, att_mask_valid_source.shape)
    print(X_valid_summary.shape, att_mask_valid_summary.shape)

output_location = base + 'BART\\'

prepare_all_data(output_location)