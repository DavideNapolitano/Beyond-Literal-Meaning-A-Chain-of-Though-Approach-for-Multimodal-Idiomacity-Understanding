import os
import httpx
import base64
import pandas as pd
import PIL.Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from ast import literal_eval

import google.generativeai as genai

# list files in /Task1/train
# print(os.getcwd())


genai.configure(api_key='')

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

split="xeval"

# MODEL_NAME="gemini-1.5-flash"
# MODEL_NAME="gemini-2.0-flash-exp"
MODEL_NAME="gemini-2.0-flash-thinking-exp-01-21"

df= pd.read_csv(f'Task1/{split}/subtask_a_{split}.tsv', delimiter='\t',encoding='utf-8')
labels_reasonings=pd.read_csv(f'Task1/{split}/labels_reasonings_G2T.csv', header=None)
labels_reasonings.columns=['Compound','Reasoning','Label', 'Gt']
# print(labels_reasonings.head())
predictions=[]
compounds=[]
print("")
if os.path.exists(f'results_CoT/{MODEL_NAME}/submission_EN.tsv') and split=="test":
    computed=pd.read_csv(f'results_CoT/{MODEL_NAME}/submission_EN.tsv', delimiter='\t',encoding='utf-8')
    computed_compounds=computed['compound'].values.tolist()
    computed_predictions=computed['expected_order'].values.tolist()
    print("Computed compounds TEST", computed_compounds)
elif os.path.exists(f'results_CoT/{MODEL_NAME}/submission_XE.tsv') and split=="xeval":
    computed=pd.read_csv(f'results_CoT/{MODEL_NAME}/submission_XE.tsv', delimiter='\t',encoding='utf-8')
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
                    f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
                ],
            }
        )
        history_idiomatic.append(
            {
                "role": "model",
                "parts": [
                    f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's an {label} compound.\n",
                ],
            }
        )
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
        q1="For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"
        response = chat_session.send_message(q1)
        # print(chat_session.history)
        ans1=response.text
        # print(ans1)

        q2="Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."
        response = chat_session.send_message(q2)
        # print(chat_session.history)
        ans2=response.text
        # print(ans2)

        q3="Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."
        response = chat_session.send_message(q3)
        # print(chat_session.history)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").strip() for el in ans3.split(",")]
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i])
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
                    f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
                ],
            }
        )
        history_literal.append(
            {
                "role": "model",
                "parts": [
                    f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's a {label} compound.\n",
                ],
            }
        )
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
        q1="For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"
        response = chat_session.send_message(q1)
        # print(chat_session.history)
        ans1=response.text
        # print(ans1)
        q2="Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one. Idiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one. Please don't use metaphors to find other meanings. Please don't do any implication. Please don't look for any potential behind what you see. You can base your decision exclusively on the Relevant Elements listed above. Do not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."
        response = chat_session.send_message(q2)
        # print(chat_session.history)
        ans2=response.text
        # print(ans2)
        q3="Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."
        response = chat_session.send_message(q3)
        # print(chat_session.history)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").strip() for el in ans3.split(",")]
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i])
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
    os.makedirs(f'results_CoT/{MODEL_NAME}', exist_ok=True)
    # os.makedirs(f'results/{MODEL_NAME}/{split}', exist_ok=True)
    if split=="test":
        df_save.to_csv(f'results_CoT/{MODEL_NAME}/submission_EN.tsv', sep='\t',index=False, header=True)
    else:
        df_save.to_csv(f'results_CoT/{MODEL_NAME}/submission_XE.tsv', sep='\t',index=False, header=True)