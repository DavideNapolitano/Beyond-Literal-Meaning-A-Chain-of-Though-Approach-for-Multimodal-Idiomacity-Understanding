import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ast import literal_eval
from sklearn.metrics import accuracy_score, dcg_score
from tqdm import tqdm

device = "cuda"
torch_dtype = torch.float16

# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/data1/hf_cache/models")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/data1/hf_cache/models")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data1/hf_cache/models")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data1/hf_cache/models")

device = "cuda"
model.to(device)

SPLIT = "test" # train, xeval, test

step_1_output = pd.read_csv(f"Task1/{SPLIT}/labels_reasonings.csv")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

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
        sentence = row["sentence"]

        print(f"Compound: {og_compound}")
        print(f"Sentence: {sentence}")
        print("Expected_order:", row["expected_order"])

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
        
        image1 = Image.open(f"Task1/{SPLIT}/{compound}/{image1_name}")
        image2 = Image.open(f"Task1/{SPLIT}/{compound}/{image2_name}")
        image3 = Image.open(f"Task1/{SPLIT}/{compound}/{image3_name}")
        image4 = Image.open(f"Task1/{SPLIT}/{compound}/{image4_name}")
        image5 = Image.open(f"Task1/{SPLIT}/{compound}/{image5_name}")

        inputs1 = processor(text=[sentence], images=image1, return_tensors="pt", padding=True)
        inputs2 = processor(text=[sentence], images=image2, return_tensors="pt", padding=True)
        inputs3 = processor(text=[sentence], images=image3, return_tensors="pt", padding=True)
        inputs4 = processor(text=[sentence], images=image4, return_tensors="pt", padding=True)
        inputs5 = processor(text=[sentence], images=image5, return_tensors="pt", padding=True)

        return {
            "inputs1": inputs1,
            "inputs2": inputs2,
            "inputs3": inputs3,
            "inputs4": inputs4,
            "inputs5": inputs5,
            "label": label,
            "rank": rank,
            "name_list": name_list,
            "og_compound": og_compound
        }

df = pd.read_csv(f"Task1/{SPLIT}/subtask_a_{SPLIT}.tsv", sep="\t")

# keep elements with subset  == "Train"
#df = df[df["subset"] == "Train"]
#df = df[df["sentence_type"] == "literal"]

# literals = step_1_output[step_1_output["label"] == "literal"]
# df = df[df["compound"].isin(literals["compound"])]

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

    inputs1 = batch["inputs1"]
    inputs2 = batch["inputs2"]
    inputs3 = batch["inputs3"]
    inputs4 = batch["inputs4"]
    inputs5 = batch["inputs5"]
    label = batch["label"]
    rank = batch["rank"]
    name_list = batch["name_list"]
    name_list = [v[0] for v in name_list]
    og_compound = batch["og_compound"]
    rank = [v.item() for v in rank]

    similarities = []

    for inputs in [inputs1, inputs2, inputs3, inputs4, inputs5]:
        input_ids = inputs["input_ids"].squeeze(1).to(device)
        attention_mask = inputs["attention_mask"].squeeze(1).to(device)
        pixel_values = inputs["pixel_values"].squeeze(1).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        text_embeds = outputs["text_embeds"]
        image_embeds = outputs["image_embeds"]

        cosine_similarity = torch.nn.functional.cosine_similarity(text_embeds, image_embeds, dim=1)
        similarities.append(cosine_similarity.item())

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

    print("NAME LIST:", name_list)

    predicted_rank_names = [name_list[i] for i in predicted_rank]
    submission_rank.append(predicted_rank_names)
    submission_compound.append(og_compound[0])

    print("SUBMISSION RANK:", predicted_rank_names)
    print("SUBMISSION COMPOUND:", submission_compound)

accuracy = accuracy_score(original_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

extended_accuracy = accuracy_score(extended_original_labels, extended_predicted_labels)
print(f"Extended Accuracy: {extended_accuracy}")

dcg = sum(dcg_scores) / len(dcg_scores)
print(f"DCG: {dcg}")

submission_file = pd.DataFrame()
submission_file["compound"] = submission_compound
submission_file["expected_order"] = submission_rank

submission_file.to_csv(f"Task1/{SPLIT}/CLIP_all_predictions.tsv", index=False, sep="\t")
