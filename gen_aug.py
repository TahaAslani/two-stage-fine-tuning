import os
import openai
import argparse
import numpy as np
import pandas as pd
from time import sleep

"""Get arguments"""
parser = argparse.ArgumentParser(description='Down sample SST-2 data')
parser.add_argument('--data-path', '-d', required=True, type=str,
    help='Directory path for the data.')
parser.add_argument('--api-key', '-k', required=True, type=str,
    help='OpenAI API Key')
args = parser.parse_args()
data_path = args.data_path
api_key = args.api_key

# Set OpenAI key:
openai.api_key = api_key

# Define genereate augmented data function:
def gen_aug(text):
    
    # The first part of prompt:
    prompt_start = prompt_start = "Rephrase the following phrase into a different phrase with similar meaning."
    
    # Full prompt:
    prompt = (prompt_start + "\n\n---\n\n" + text)

    response = ""

    # Set flag to true
    flag = True
    len_cut = int(len(prompt)*0.1)

    while flag:

        try:
            result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
            flag = False

        except:
            prompt = prompt[:-len_cut]

    response = result["choices"][0]["message"]["content"]
    
    return response

# If we are continuing augmented data generation:
if os.path.isfile(os.path.join(data_path,'tmp.csv')):

    # Load data
    df = pd.read_csv(os.path.join(data_path,'tmp.csv'))
    data_count = pd.read_csv(os.path.join(data_path,'data_count.csv'))

# If we start generating data augmentation:
else:

    # Initialize data
    df = pd.read_csv(os.path.join(data_path,'train.csv'))
    df['num_aug_gen'] = 0

    data_count = pd.DataFrame()
    for label in [0, 1]:
        data_count.loc[label, 'original'] = int(df['label'].value_counts()[label])
    data_count['aug'] = 0

n_aug = int(np.max(data_count['original']))

error_count = 0

# While there is a class that needs more data
while np.sum(data_count['aug']<n_aug*1.01)>0:
    
    for i in df.index:

        l = df.loc[i,'label']

        if data_count.loc[l, 'aug']>n_aug*1.01:
            continue


        print(i,'/',len(df))

        text = df.loc[i,'text']

        # Delay to bypass rate limitation
        sleep(1)


        try:

            aug_texts = gen_aug(text)

            j =  df.loc[i,'num_aug_gen']
            df.loc[i,'tmp_aug_'+str(j)]=aug_texts

            data_count.loc[l, 'aug']  = data_count.loc[l, 'aug'] + 1
            df.loc[i,'num_aug_gen'] = df.loc[i,'num_aug_gen'] + 1

            print(l, data_count.loc[l, 'aug'])

        except:

            sleep(60)

            error_count = error_count + 1
    
            if error_count>5:
                raise Exception('Too many errors')

        df.to_csv(os.path.join(data_path,'tmp.csv'), index=False)
        data_count.to_csv(os.path.join(data_path,'data_count.csv'), index=False)






aug_cols = []

for col in df.columns:
    
    if col.startswith('tmp_aug'):

       	aug_cols.append(col)

NaNs = df.isna()

all_data = pd.DataFrame(columns=['org_ind', 'paragraph', 'label',])

counter = 0

for i in df.index:
    
    l = df.loc[i, 'label']
    
    for col in aug_cols:

       	if not NaNs.loc[i, col]:

            all_data.loc[counter, 'org_ind'] = i
            all_data.loc[counter, 'paragraph'] = df.loc[i, col]
            all_data.loc[counter, 'label'] = int(l)

            counter += 1

out = pd.DataFrame(columns=['org_ind', 'paragraph', 'label',])

for l in set(all_data['label']):
    
    data_label = all_data.loc[all_data['label']==l, :]
    
    data_label_sampled = data_label.sample(n=n_aug, random_state=42)
    
    out = pd.concat([out, data_label_sampled])
    

out.rename(columns={'paragraph': "text"}, inplace=True)

out_shuff = out.sample(frac=1, random_state=42).reset_index(drop=True)

out_shuff.to_csv(os.path.join(data_path,'aug.csv'), index=False)
