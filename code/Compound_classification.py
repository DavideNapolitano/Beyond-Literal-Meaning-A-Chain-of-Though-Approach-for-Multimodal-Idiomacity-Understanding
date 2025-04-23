import os
import httpx
import base64
import pandas as pd
import PIL.Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
from ast import literal_eval
import time

import google.generativeai as genai

# list files in /Task1/train
print(os.getcwd())



genai.configure(api_key='')

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 65536,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
#   model_name="gemini-2.0-flash-thinking-exp-01-21",
#   model_name="gemini-1.5-flash",
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
)

tot=0
correct=0

top1=0
labels=[]
reasonings=[]
compounds=[]
gts=[]
split='xeval'
df= pd.read_csv(f'Task1/{split}/subtask_a_{split}.tsv', delimiter='\t',encoding='utf-8')
coompounds_list=df['compound'].values.tolist()
idx=1
for compound in coompounds_list:

    print("-"*50)
    print(idx, compound)
    idx+=1
    time.sleep(7)

    if compound == 'cat_s eyes':
        compound = "cat's eyes"
    if compound == 'dog_s dinner':
        compound = "dog's dinner"
    if compound == 'devil_s advocate':
        compound = "devil's advocate"
    if compound == 'pig_s ear':
        compound = "pig's ear"

    sentence = df[df['compound']==compound]['sentence'].values[0]
    if split == 'train':
        sentence_type = df[df['compound']==compound]['sentence_type'].values[0]
    # sentence_type = df[df['compound']==compound]['sentence_type'].values[0]
    # image1_caption = df[df['compound']==compound]['image1_caption'].values[0]
    # image2_caption = df[df['compound']==compound]['image2_caption'].values[0]
    # image3_caption = df[df['compound']==compound]['image3_caption'].values[0]
    # image4_caption = df[df['compound']==compound]['image4_caption'].values[0]
    # image5_caption = df[df['compound']==compound]['image5_caption'].values[0]
    # expected_order = df[df['compound']==compound]['expected_order'].values[0]
    # expected_order = literal_eval(expected_order)
    # print(expected_order, type(expected_order))
    print(sentence)

    prompt1 = f"""
    I'm working on project where I have a set of 5 images and a context sentence in which a particular potentially idiomatic nominal compound (NC) appears. 
    The goal is to rank the images according to how well they represent the sense in which the NC is used in the given context sentence.

    The first step is to classify if the setting where the compound is uses is leteral or idiomatic.
    The classification is done given the compound name and the associated sentence.
    Please help me in the classification by returning the REASONING and the LABEL between "idiomatic" or "literal"

    COMPOUND:
    {compound}

    SENTENCE:
    {sentence}

    Use this output format:
    REASONING: xxxxx

    LABEL: xxxxx
    """
    response = model.generate_content(prompt1)
    ans = response.text.strip()
    # print(sentence_type)
    # print(ans)

    elems=ans.split('\n')
    # print(elems)
    reason=elems[0].split(':')[-1].strip()
    lbl=elems[-1].split(':')[-1].strip()
    print("LABEL:",lbl)
    print("REASONING:",reason)
    reasonings.append(reason)
    labels.append(lbl)
    compounds.append(compound)
    gts.append("")

    if split == 'train':
        if sentence_type == lbl:
            correct+=1
        tot+=1

if split == 'train':
    print(f"Accuracy Sentence Classification: {correct/tot}")
df=pd.DataFrame({'compound':compounds, 'reasoning':reasonings, 'label':labels, 'gt':gts})
df.to_csv(f'Task1/{split}/labels_reasonings_G2T.csv', index=False)
