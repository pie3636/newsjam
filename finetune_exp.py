from transformers import AutoTokenizer, AutoModel
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

x_train = list(x_train)
x_test = list(x_test)

#%%

# Tokenize train and test set

train_encodings = tokenizer(x_train, truncation=True, padding=True)
test_encodings = tokenizer(x_test, truncation=True, padding=True)
#%%
print(len(train_encodings['input_ids']))
print(len(test_encodings['input_ids']))

# Organize train and test data back into one dictionary (not really necessary, kind of redundant after splitting data earlier)
'''
tokenized_data = {}
tokenized_data['train'] = train_tokenized
tokenized_data['test'] = test_tokenized
'''
# Instantiate data collator
#data_collator = DataCollatorWithPadding(tokenizer)

#%%
import tensorflow as tf

# Transforming encodings into tensorflow Dataset objects

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings)
    ))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings)
    ))

#%%
from transformers import TFTrainingArguments, TFTrainer

# Instantiate training arguments (currently the output path is the only parameter I've entered, but more specific training arguments can be entered)
training_args = TFTrainingArguments(
    output_dir = "~/Desktop/finetune_test_trainer",
    num_train_epochs=2,
    max_steps=100,
                                  )

# Instantiate trainer
trainer = TFTrainer(model=model, 
                  args=training_args, 
                  train_dataset=train_dataset, 
                  eval_dataset=test_dataset,
                  )

# Train the model on the new dataset
trainer.train()
