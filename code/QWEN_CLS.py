import os
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# list files in /Task1/train
print(os.getcwd())

SPLIT = "xeval" # train, xeval, test
step_1_output = pd.read_csv(f"Task1/{SPLIT}/labels_reasonings.csv")
df = pd.read_csv(f"Task1/{SPLIT}/subtask_a_{SPLIT}.tsv", sep="\t")
labels_reasonings=pd.read_csv(f'Task1/{SPLIT}/labels_reasonings_G2T.csv')
print(labels_reasonings.columns)
# idiomatic = step_1_output[step_1_output["label"] == "idiomatic"]
# df = df[df["compound"].isin(idiomatic["compound"])]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto",cache_dir="/data1/hf_cache/models")
min_pixels = 256 * 28 * 28
max_pixels = 256 * 28 * 28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

submission_compound = []
submission_rank = []
submission_generated_text = []
submission_image_list = []

for compound in tqdm(list(df['compound'])):
    submission_compound.append(compound)
    folder_name = compound.replace("'", "_")
    list_images=[]
    just_images=[]
    for image in os.listdir(f'Task1/{SPLIT}/'+folder_name):
        list_images.append(f'Task1/{SPLIT}/'+folder_name+'/'+image)
        just_images.append(image)

    row = labels_reasonings[labels_reasonings['compound']==compound]
    label = row['label'].values[0]
    reasoning = row['reasoning'].values[0]
    sentence = df[df['compound']==compound]['sentence'].values[0]
    #expected_order = df[df['compound']==compound]['expected_order']

    history = []
    # print("\n____________")
    # print("COMPOUND:", compound)
    # print("LABEL:", label)
    # print("REASONING:", reasoning)
    # print("SENTENCE:", sentence)

    if label=='idiomatic':

        prompt = f"""
Given a compound, a sentence, and five different images, rank the images according to the meaning of the compound in the sentence
In this case, the compound is used as an idiomatic expression in the sentence. 
In detail, rank the images from the most pertinent to the least one. Irrelevant images must be placed in last position.

Compound:
{compound}

Sentence:
{sentence}

Provide a rank including all images, do not provide any explanation. Use the following format:
- IMAGE 1
- IMAGE 2
- IMAGE 3
- IMAGE 4
- IMAGE 5
"""
    else:
        prompt = f"""
Given a compound, a sentence, and five different images, rank the images according to the meaning of the compound in the sentence
In this case, the compound is used as a literal expression in the sentence.
In detail, rank the images from the most pertinent to the least one. Irrelevant images must be placed in last position.

Compound:
{compound}

Sentence:
{sentence}

Provide a rank including all images, do not provide any explanation. Use the following format:
- IMAGE 1
- IMAGE 2
- IMAGE 3
- IMAGE 4
- IMAGE 5
"""

    content= []
    for img in list_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    history.append({"role": "user", "content": content})

    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("OUTPUT TEXT:", output_text[0])

    submission_generated_text.append(output_text[0])
    submission_image_list.append(just_images)

submission_file = pd.DataFrame()
submission_file["compound"] = submission_compound
submission_file["image_list"] = submission_image_list
submission_file["generated_text"] = submission_generated_text

submission_file.to_csv(f"Task1/{SPLIT}/Qwen_CLS_all_predictions.csv", index=False)


# CAPTIONS:
# IMAGE1:{image1_caption}
# IMAGE2:{image2_caption}
# IMAGE3:{image3_caption}
# IMAGE4:{image4_caption}
# IMAGE5:{image5_caption}