#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:04:38 2022

@author: josephkeenan
"""

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import pandas as pd

# Instantiate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

# Load data
data = pd.read_json('~/Downloads/guardian_data.json')


#%%

from sklearn.model_selection import train_test_split

# Split data into train and test set
x_train, x_test = train_test_split(data['text'])


#%%

# Tokenize train and test set
train_tokenized = tokenizer([x for x in x_train], truncation=True)
test_tokenized = tokenizer([x for x in x_test], truncation=True)

# Organize train and test data back into one dictionary (not really necessary, kind of redundant after splitting data earlier)
tokenized_data = {}
tokenized_data['train'] = train_tokenized
tokenized_data['test'] = test_tokenized

# Instantiate data collator
data_collator = DataCollatorWithPadding(tokenizer)

#%%

#%%
from transformers import TrainingArguments, Trainer

# Instantiate training arguments (currently the output path is the only parameter I've entered, but more specific training arguments can be entered)
training_args = TrainingArguments("~/Documents/S7_M1_TAL/UE703_Software_Engineering/703_Project/test_trainer")

# Instantiate trainer
trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=tokenized_data['train']['input_ids'], 
                  eval_dataset=tokenized_data['test']['input_ids'], 
                  data_collator=data_collator, 
                  tokenizer=tokenizer)

# Train the model on the new dataset
trainer.train()





