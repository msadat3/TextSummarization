import os
from Utils import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import os.path as p
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer
from Utils import *

model_type = "BART"
prertained_model_name = 'facebook/bart-base'
base = "/home/ubuntu/CNNDM/" + model_type + "/"
model_loc = base +'/checkpoints/best_checkpoint.pt'
output_file_location = base + 'generated_summaries.txt'

if model_type == "BART":
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained(prertained_model_name)
model.load_state_dict(torch.load(model_loc))
model.cuda()

test_sources = load_data(base + 'X_test_source.pkl')
test_sources_att = load_data(base + 'att_mask_test_source.pkl')
#test_summary = load_data(base + 'X_valid_summary.pkl')
#test_summary_att = load_data(base + 'att_mask_valid_summary.pkl')
generated_list = []
i = 0
with open(output_file_location,'w') as output_file:
    for t_source, t_att in zip(test_sources, test_sources_att):
        t_source = torch.as_tensor(t_source)
        t_att = torch.as_tensor(t_att)
        t_source = t_source.to(torch.device('cuda')).unsqueeze(0)
        t_att = t_att.to(torch.device('cuda')).unsqueeze(0)

        #print(t_source, type(t_source))
        summary_ids = model.generate(t_source,attention_mask=t_att, num_beams=4, max_length=1024, early_stopping=True)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        print(i, summary[0])
        output_file.write(summary[0]+'\n')
        generated_list.append(summary[0])
        i+=1
save_data(generated_list,base + 'generated_summaries_list.pkl')

#####prepare reference summaries in one file:

 #summary_ids = model.generate(inputs['input_ids'].to(torch.device('cuda')),attention_mask=inputs['attention_mask'].to(torch.device('cuda')), num_beams=4, max_length=5, early_stopping=True)




