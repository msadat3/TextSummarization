import os
from Utils import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import os.path as p
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


model_type = 'PEGASAS'
prertained_model_name = 'google/pegasus-large'

base = "/home/ubuntu/Keyphrase_Generation/DataForExperiments_BART/"+model_type

checkpoint_location = base + '/checkpoints/'

if p.exists(checkpoint_location) == False:
    os.mkdir(checkpoint_location)

def create_data_loaders(X, X_att_mask, y, y_att_mask, batch_size,data_type='train'):
    X = torch.tensor(X, dtype=torch.long)
    X_att_mask = torch.tensor(X_att_mask, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    y_att_mask = torch.tensor(y_att_mask, dtype=torch.long)

    data = TensorDataset(X, X_att_mask, y, y_att_mask)
    if data_type != 'train':
        # data = sorted(data, key=lambda x: x[3], reverse=True)#sorting by target length in descending order
        # return data
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader

batch_size = 8
accumulation_steps = int(256 / batch_size)
learning_rate = 2e-5
num_epochs = 3
model_dim = 1024
warmup_steps = 4000

report_every = 1
validation_every = 10000
device = 'cuda'
last_checkpoint_info = {}
best_checkpoint_info = {}

last_checkpoint_info['epoch'] = 0
last_checkpoint_info['step_count'] = 0
last_checkpoint_info['i'] = 0
last_checkpoint_info['current_best_perplexity'] = 999999999


last_checkpoint_info_location = checkpoint_location + 'last_checkpoint_info.pkl'
best_checkpoint_info_location = checkpoint_location + 'best_checkpoint_info.pkl'
last_checkpoint_location = checkpoint_location + 'last_checkpoint.pt'
best_checkpoint_location = checkpoint_location + 'best_checkpoint.pt'

def train_model(train_data_loader, validation_data_loader):

    if model_type == 'PEGASAS':
        model = PegasusForConditionalGeneration.from_pretrained(prertained_model_name)
        tokenizer = PegasusTokenizer.from_pretrained(prertained_model_name)
        if p.exists(last_checkpoint_info_location) == True:
            model.load_state_dict(torch.load(last_checkpoint_location))
            last_checkpoint_info = load_data(last_checkpoint_info_location)

    if device == 'cuda':
        model.cuda()
    lr = learning_rate

    optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    prev_validation_perplexity = last_checkpoint_info['current_best_perplexity']
    patience = 5
    not_improving_checkpoints = 0
    train_stop_flag = False

    for epoch in range(last_checkpoint_info['epoch'], num_epochs):
        model.train()
        step_count = 0
        i=0
        optimizer.zero_grad()
        for X, X_att_mask, y, y_att_mask in train_data_loader:
            while step_count!=last_checkpoint_info['epoch'] or i!= last_checkpoint_info['i']:
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader) == 0:
                    step_count+=1
                i += 1
                continue
            input_ids = X.to(torch.device(device))
            attention_mask = X_att_mask.to(torch.device(device))
            y = y.to(torch.device(device))
            y_att_mask = y_att_mask.to(torch.device(device))

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_attention_mask=y_att_mask, labels=y, return_dict=False)

            crossEntropyLoss = outputs[0]
            loss = crossEntropyLoss/accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader):
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                print('Epoch', epoch, 'step', step_count, 'loss', loss.item(), 'current validation perplexity',
                      prev_validation_perplexity)

                if step_count % validation_every == 0 or ((i + 1) == len(train_data_loader)) and step_count != 0:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = 0
                        batch_count = 0

                        for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))
                            val_y = val_y.to(torch.device(device))
                            val_y_att_mask = val_y_att_mask.to(torch.device(device))

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                decoder_attention_mask=val_y_att_mask,
                                                labels=val_y,return_dict=False)

                            val_loss_batch = val_outputs[0]
                            validation_loss += val_loss_batch.item()
                        validation_perplexity = math.exp(validation_loss / batch_count)
                        print('perp', validation_perplexity)
                        last_checkpoint_info['epoch'] = epoch
                        last_checkpoint_info['step_count'] = step_count
                        last_checkpoint_info['i'] = i

                        if validation_perplexity < prev_validation_perplexity:
                            print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  validation_perplexity)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                            last_checkpoint_info['current_best_perplexity'] = validation_perplexity
                            torch.save(model.state_dict(), checkpoint_location+'/last_ckeckpoint.pt')
                            torch.save(model.state_dict(), checkpoint_location + '/best_ckeckpoint.pt')
                            save_data(last_checkpoint_info,last_checkpoint_info_location)
                            best_checkpoint_info = last_checkpoint_info
                            save_data(best_checkpoint_info, best_checkpoint_info_location)
                        else:
                            print("Validation perplexity did not improve.")
                            not_improving_checkpoints += 1
                            torch.save(model.state_dict(), last_checkpoint_location)
                            print(last_checkpoint_info)
                            save_data(last_checkpoint_info, last_checkpoint_info_location)
                        model.train()
                    if not_improving_checkpoints == patience:
                        print("Not improving for ", patience, " checkpoints. Sopping training.")
                        train_stop_flag = True
                        break
            i+=1
        if train_stop_flag == True:
            break