import os
import httpx
import base64
import pandas as pd
import PIL.Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from ast import literal_eval

import google.generativeai as genai

genai.configure(api_key='')

# split="test"
split="xeval"

# MODEL_NAME="gemini-1.5-flash"
# MODEL_NAME="gemini-2.0-flash-exp"
MODEL_NAME="gemini-2.0-flash-thinking-exp-01-21"

print(MODEL_NAME, '-', split)

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model
generation_config = {
  "temperature": 0.7, #0.7
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}


def read_images(compounds):
    files=[]
    for compound in compounds:
        for image in os.listdir('Task1/train/'+compound):
            files.append(upload_to_gemini('Task1/train/'+compound+'/'+image, mime_type="image/png"))
    return files


history_idiomatic_MAIN=[]
history_literal_MAIN=[]

tot=0
correct=0

top1=0

df= pd.read_csv(f'Task1/{split}/subtask_a_{split}.tsv', delimiter='\t',encoding='utf-8')
labels_reasonings=pd.read_csv(f'Task1/{split}/labels_reasonings_G2T.csv', header=None)
labels_reasonings.columns=['Compound','Reasoning','Label', 'Gt']
predictions=[]
compounds=[]
print("")
if os.path.exists(f'results_plain/{MODEL_NAME}/submission_EN.tsv') and split=="test":
    computed=pd.read_csv(f'results_plain/{MODEL_NAME}/submission_EN.tsv', delimiter='\t',encoding='utf-8')
    computed_compounds=computed['compound'].values.tolist()
    computed_predictions=computed['expected_order'].values.tolist()
    print("Computed compounds TEST", computed_compounds)
elif os.path.exists(f'results_plain/{MODEL_NAME}/submission_XE.tsv') and split=="xeval":
    computed=pd.read_csv(f'results_plain/{MODEL_NAME}/submission_XE.tsv', delimiter='\t',encoding='utf-8')
    computed_compounds=computed['compound'].values.tolist()
    computed_predictions=computed['expected_order'].values.tolist()
    print("Computed compounds XEVAL", computed_compounds)
else:
    print("No computed compounds")
    computed_compounds=[]
    computed_predictions=[]

idx=1
coompounds_list=df['compound'].values.tolist()

for compound in coompounds_list:

    print("-"*50)
    print(idx, compound)
    idx+=1

    just_images=[]
    for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
        just_images.append(image)

    association={}
    for i,el in enumerate(just_images):
        # association_string+=f"IMAGE{i+1}: {el}\n"
        association[f"Image{i+1}"]=el

    
    if compound == 'cat_s eyes':
        compound = "cat's eyes"
    if compound == 'dog_s dinner':
        compound = "dog's dinner"
    if compound == 'devil_s advocate':
        compound = "devil's advocate"
    if compound == 'pig_s ear':
        compound = "pig's ear"

    
    if compound in computed_compounds:
        print("Already computed")
        continue
    compounds.append(compound)

    try:
        row = labels_reasonings[labels_reasonings['Compound']==compound]
        label = row['Label'].values[0]
        reasoning = row['Reasoning'].values[0]
    except:
        print(compound)
        print(row)

    sentence = df[df['compound']==compound]['sentence'].values[0]
    image1_caption = df[df['compound']==compound]['image1_caption'].values[0]
    image2_caption = df[df['compound']==compound]['image2_caption'].values[0]
    image3_caption = df[df['compound']==compound]['image3_caption'].values[0]
    image4_caption = df[df['compound']==compound]['image4_caption'].values[0]
    image5_caption = df[df['compound']==compound]['image5_caption'].values[0]
    print("P:", label)

    history_idiomatic=history_idiomatic_MAIN.copy()
    history_literal=history_literal_MAIN.copy()

    if len(history_idiomatic)!=0 or len(history_literal)!=0:
        print("History not empty")
        exit()

    # print(expected_order)
    if label=='idiomatic':
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
        files2=[]
        for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
            files2.append(upload_to_gemini(f'Task1/{split}/'+compound.replace("'","_")+'/'+image, mime_type="image/png"))
        # print(expected_order)

        history_idiomatic.append(
            {
                "role": "user",
                "parts": [
                    files2[0],
                    files2[1],
                    files2[2],
                    files2[3],
                    files2[4],
                ],
            }
        )
        chat_session = model.start_chat(history=history_idiomatic)
        # q1="For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"
        q1=f""""
        Given a compound, a sentence, and five different images, rank the images according to the meaning of the compound in the sentence. 
        Consider that the compound can be used literally or as an idiomatic expression in the sentence. 
        In detail, rank the images from the most pertinent to the least one. Irrelevant images must be placed in last position.
        Provide a rank including all images, do not provide any explanation.
        
        Return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5]. Images goes from 1 (Image1) to 5 (Image5). 
        Return ONLY the list.

        Compound:
        {compound}

        Sentence:
        {sentence}
        """""
        response = chat_session.send_message(q1)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").replace("'","").strip() for el in ans3.split(",")]
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i.replace("\n","").replace("`","")])
        # print(pred)
        predictions.append(pred)
        # break
        # if pred[0]==expected_order[0]:
        #     correct+=1
        # tot+=1
    else:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
        files2=[]
        for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
            files2.append(upload_to_gemini(f'Task1/{split}/'+compound.replace("'","_")+'/'+image, mime_type="image/png"))
        # print(expected_order)

        history_literal.append(
            {
                "role": "user",
                "parts": [
                    files2[0],
                    files2[1],
                    files2[2],
                    files2[3],
                    files2[4],
                ],
            }
        )
        chat_session = model.start_chat(history=history_literal)
        q1=f""""
        Given a compound, a sentence, and five different images, rank the images according to the meaning of the compound in the sentence. 
        Consider that the compound can be used literally or as an idiomatic expression in the sentence. 
        In detail, rank the images from the most pertinent to the least one. Irrelevant images must be placed in last position.
        Provide a rank including all images, do not provide any explanation.

        Return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5]. Images goes from 1 (Image1) to 5 (Image5). 
        Return ONLY the list.

        Compound:
        {compound}

        Sentence:
        {sentence}
        """""
        response = chat_session.send_message(q1)
        # print(chat_session.history)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").replace("'","").strip() for el in ans3.split(",")]
        print(ans3_list)
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i.replace("\n","").replace("`","")])
        print(pred)
        predictions.append(pred)
        # break
        # if pred[0]==expected_order[0]:
        #     correct+=1
        # tot+=1

    # print("Accuracy:", correct/tot)    
    computed_predictions.append(pred)
    computed_compounds.append(compound)
    df_save=pd.DataFrame({'compound':computed_compounds,'expected_order':computed_predictions})
    os.makedirs(f'results_plain/{MODEL_NAME}', exist_ok=True)
    # os.makedirs(f'results/{MODEL_NAME}/{split}', exist_ok=True)
    if split=="test":
        df_save.to_csv(f'results_plain/{MODEL_NAME}/submission_EN.tsv', sep='\t',index=False, header=True)
    else:
        df_save.to_csv(f'results_plain/{MODEL_NAME}/submission_XE.tsv', sep='\t',index=False, header=True)