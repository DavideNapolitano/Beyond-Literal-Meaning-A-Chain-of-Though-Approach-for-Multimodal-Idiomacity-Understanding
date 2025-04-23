import pandas as pd
import torch
from PIL import Image
from transformers import CLIPTextModel, AutoTokenizer
from ast import literal_eval
from sklearn.metrics import accuracy_score, dcg_score
from tqdm import tqdm

device = "cuda"
torch_dtype = torch.float16

processor = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data1/hf_cache/models")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data1/hf_cache/models")

device = "cuda"
model.to(device)

SPLIT = "xeval" # train, xeval, test

step_1_output = pd.read_csv(f"Task1/{SPLIT}/labels_reasonings.csv")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        og_compound = row["compound"]
        compound = og_compound.replace("'", "_")
        image1_name = row["image1_name"]
        image2_name = row["image2_name"]
        image3_name = row["image3_name"]
        image4_name = row["image4_name"]
        image5_name = row["image5_name"]
        image1_caption = row["image1_caption"]
        image2_caption = row["image2_caption"]
        image3_caption = row["image3_caption"]
        image4_caption = row["image4_caption"]
        image5_caption = row["image5_caption"]
        sentence = row["sentence"]

        # print(f"Compound: {og_compound}")
        # print(f"Sentence: {sentence}")
        # print("Expected_order:", row["expected_order"])

        if SPLIT == "train":
            expected_order = row["expected_order"]
            expected_order = literal_eval(expected_order)
            reference = expected_order[0]
            if reference == image1_name:
                label = 0
            elif reference == image2_name:
                label = 1
            elif reference == image3_name:
                label = 2
            elif reference == image4_name:
                label = 3
            elif reference == image5_name:
                label = 4
            else:
                raise ValueError("Reference not found in the images")
        
            rank_dict = {
                image1_name: 0,
                image2_name: 1,
                image3_name: 2,
                image4_name: 3,
                image5_name: 4
            }
            rank = [rank_dict[el] for el in expected_order]
        else:
            label = -1
            rank = [-1, -1, -1, -1, -1]
        
        name_list = [str(image1_name), str(image2_name), str(image3_name), str(image4_name), str(image5_name)]
        
        inputs_sentence = self.tokenizer(text=[sentence], return_tensors="pt", padding=True, truncation=True)
        inputs_captions = self.tokenizer(text=[image1_caption, image2_caption, image3_caption, image4_caption, image5_caption], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "inputs_sentence": inputs_sentence,
            "inputs_captions": inputs_captions,
            "label": label,
            "rank": rank,
            "name_list": name_list,
            "og_compound": og_compound
        }

df = pd.read_csv(f"Task1/{SPLIT}/subtask_a_{SPLIT}.tsv", sep="\t")

dataset = CustomDataset(df, processor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

predicted_labels = []
original_labels = []

extended_predicted_labels = []
extended_original_labels = []

dcg_scores = []
submission_rank = []
submission_compound = []
for batch in tqdm(dataloader):

    inputs_sentence = batch["inputs_sentence"]
    inputs_captions = batch["inputs_captions"]
    
    label = batch["label"]
    rank = batch["rank"]
    name_list = batch["name_list"]
    name_list = [v[0] for v in name_list]
    og_compound = batch["og_compound"]
    rank = [v.item() for v in rank]

    similarities = []


    sententence_input_ids = inputs_sentence["input_ids"].squeeze(1).to(device)
    sentence_attention_mask = inputs_sentence["attention_mask"].squeeze(1).to(device)
    outputs_sentence = model(input_ids=sententence_input_ids, attention_mask=sentence_attention_mask)
    sentence_embedding = outputs_sentence.pooler_output

    caption_input_ids = inputs_captions["input_ids"].squeeze(1).squeeze(0).to(device)
    caption_attention_mask = inputs_captions["attention_mask"].squeeze(1).squeeze(0).to(device)
    outputs_captions = model(input_ids=caption_input_ids, attention_mask=caption_attention_mask)
    caption_embeddings = outputs_captions.pooler_output

    for caption_embedding in caption_embeddings:
        caption_embedding = caption_embedding.unsqueeze(0)
        similarity = torch.cosine_similarity(sentence_embedding, caption_embedding, dim=1)
        similarities.append(similarity.item())    

    similarities = torch.tensor(similarities)
    predicted_rank = torch.argsort(similarities, descending=True)

    extended_predicted_labels.extend(predicted_rank)
    extended_original_labels.extend(rank)

    # print(f"Compound: {og_compound}")
    # print(f"Original rank: {rank}")
    # print(f"Predicted rank: {predicted_rank}")

    predicted_label = torch.argmax(similarities, dim=0)
    predicted_labels.append(predicted_label.item())
    original_labels.append(label.item())

    # print(f"Original label: {label}")
    # print(f"Predicted label: {predicted_label}")

    dcg = dcg_score([rank], [predicted_rank])
    dcg_scores.append(dcg)

    # print("NAME LIST:", name_list)

    predicted_rank_names = [name_list[i] for i in predicted_rank]
    submission_rank.append(predicted_rank_names)
    submission_compound.append(og_compound[0])

    # print("SUBMISSION RANK:", predicted_rank_names)
    # print("SUBMISSION COMPOUND:", submission_compound)

accuracy = accuracy_score(original_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

extended_accuracy = accuracy_score(extended_original_labels, extended_predicted_labels)
print(f"Extended Accuracy: {extended_accuracy}")

dcg = sum(dcg_scores) / len(dcg_scores)
print(f"DCG: {dcg}")

submission_file = pd.DataFrame()
submission_file["compound"] = submission_compound
submission_file["expected_order"] = submission_rank

submission_file.to_csv(f"Task1/{SPLIT}/CLIP-text_all_predictions.tsv", index=False, sep="\t")
