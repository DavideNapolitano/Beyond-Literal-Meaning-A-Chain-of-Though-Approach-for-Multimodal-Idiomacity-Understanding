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

SPLIT = "test" # train, xeval, test
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

'''
for compound in tqdm(list(df['compound'])):
    submission_compound.append(compound)
    folder_name = compound.replace("'", "_")
    list_images=[]
    just_images=[]
    for image in os.listdir(f'Task1/{SPLIT}/'+folder_name):
        # print(image)
        list_images.append(f'Task1/{SPLIT}/'+folder_name+'/'+image)
        just_images.append(image)
    
    content_1 = []

    sentence = df[df['compound']==compound]['sentence'].values[0]
    expected_order = df[df['compound']==compound]['expected_order']

    image1_caption = df[df['compound']==compound]['image1_caption'].values[0]
    image2_caption = df[df['compound']==compound]['image2_caption'].values[0]
    image3_caption = df[df['compound']==compound]['image3_caption'].values[0]
    image4_caption = df[df['compound']==compound]['image4_caption'].values[0]
    image5_caption = df[df['compound']==compound]['image5_caption'].values[0]
    
    prompt_1=f"""
    I'm working on project where I have a set of five images and a context sentence in which an idiomatic compound appears.
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

    content_1.append({"type": "text", "text": prompt_1})

    content_assistant = step_1_output[step_1_output["compound"] == compound]["reasoning"].values[0]

    content_2= []
    for img in list_images:
        content_2.append({"type": "image", "image": img})

    prompt_2=f"""
    Now, I'll provide five images.
    For each image, extract the key elements with respect to the compound and the reasoning about its meaning. 
    Provide also a score between 1 and 100, where 1 means the image does not represent the compound and 100 means the image represents the ideomatic expression of the compound. 
    Do not assign equal scores. Use the following format:
    IMAGE1: [image1_statement] - [image1_score]
    IMAGE2: [image2_statement] - [image2_score]
    IMAGE3: [image3_statement] - [image3_score]
    IMAGE4: [image4_statement] - [image4_score]
    IMAGE5: [image5_statement] - [image5_score]
    """

    # print(prompt)
    content_2.append({"type": "text", "text": prompt_2})

    # Messages containing multiple images and a text query
    messages = [
        {
            "role": "user",
            "content": content_1,
        },
        {
            "role": "assistant",
            "content": content_assistant,
        },
        {
            "role": "user",
            "content": content_2,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("COMPUOND:", compound)
    print("OUTPUT TEXT:", output_text[0])

    # # Extracting the rank
    # rank = output_text[0].split("\n")
    # scores = [int(r.split("-")[-1].strip()) for r in rank]
    # rank = list(range(1, 6))
    # rank = [just_images[x-1] for _, x in sorted(zip(scores, rank), reverse=True)]
    # print("JUST IMAGES", just_images)
    # print("PREDICTED RANK:", rank)
    # submission_rank.append(rank)

    submission_generated_text.append(output_text[0])
    submission_image_list.append(just_images)
'''

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

        # print("IDIOMATIC")
        content = []
        content.append({"type": "text", "text": f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n"})
        history.append({"role": "user", "content": content})

        content = []
        content.append({"type": "text", "text": f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's an {label} compound.\n"})
        history.append({"role": "assistant", "content": content})

        content= []
        for img in list_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 1:", output_text[0])

        content = []
        content.append({"type": "text", "text": output_text[0]})
        history.append({"role": "assistant", "content": content})

        content = []
        content.append({"type": "text", "text": "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 2:", output_text[0])

        content = []
        content.append({"type": "text", "text": output_text[0]})
        history.append({"role": "assistant", "content": content})

        content = []
        content.append({"type": "text", "text": "Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 3:", output_text[0])

    else:
        # print("LITERAL")
        content = []
        content.append({"type": "text", "text": f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n"})
        history.append({"role": "user", "content": content})

        content = []
        content.append({"type": "text", "text": f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's an {label} compound.\n"})
        history.append({"role": "assistant", "content": content})

        content= []
        for img in list_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 1:", output_text[0])

        content = []
        content.append({"type": "text", "text": output_text[0]})
        history.append({"role": "assistant", "content": content})

        content = []
        content.append({"type": "text", "text": "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one. Idiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one. Please don't use metaphors to find other meanings. Please don't do any implication. Please don't look for any potential behind what you see. You can base your decision exclusively on the Relevant Elements listed above. Do not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 2:", output_text[0])

        content = []
        content.append({"type": "text", "text": output_text[0]})
        history.append({"role": "assistant", "content": content})

        content = []
        content.append({"type": "text", "text": "Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. The list must include only the image names, like for example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."})
        history.append({"role": "user", "content": content})

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("STEP 3:", output_text[0])

    submission_generated_text.append(output_text[0])
    submission_image_list.append(just_images)

submission_file = pd.DataFrame()
submission_file["compound"] = submission_compound
submission_file["image_list"] = submission_image_list
submission_file["generated_text"] = submission_generated_text

submission_file.to_csv(f"Task1/{SPLIT}/Qwen_CoT_all_predictions.csv", index=False)


# CAPTIONS:
# IMAGE1:{image1_caption}
# IMAGE2:{image2_caption}
# IMAGE3:{image3_caption}
# IMAGE4:{image4_caption}
# IMAGE5:{image5_caption}