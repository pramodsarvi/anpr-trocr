import os
import pandas as pd
ROOT = ""
BATCH_SIZE = 48
EPOCHS = 20
df = pd.read_csv(ROOT+'good.csv', header=None)
df.reset_index(drop=True, inplace=True) 
df = df.drop(columns=[0],axis=0)
df.rename(columns={1: "file_name", 2: "text"}, inplace=True)
df = df.drop(0)
# df.head()

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

from torchvision import transforms
T=transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=.60, hue=.15,contrast=0.6),
            transforms.RandomRotation(degrees=(340, 390)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            transforms.RandomAutocontrast(p=0.45),
])


import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        # print(file_name)
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        image = T(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1",use_fast=False)
train_dataset = IAMDataset(root_dir=ROOT,
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=ROOT,
                           df=test_df,
                           processor=processor)


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)


from transformers import VisionEncoderDecoderModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("first_run/15",local_files_only=True)
model.to(device)


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


from datasets import load_metric

cer_metric = load_metric("cer")


def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


from transformers import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=1e-4)


from accelerate import Accelerator
accelerator = Accelerator()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0, last_epoch=-1, verbose='deprecated')
model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

best_cer = 9999

for epoch in range(EPOCHS):  # loop over the dataset multiple times
#    # train
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        # get the inputs
        for k,v in batch.items():
            batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            train_loss += loss.item()

    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

    # evaluate
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to(device))
            # compute metrics
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer

    current_cer = valid_cer / len(eval_dataloader)
    print("Validation CER:", current_cer)

    if current_cer < best_cer:
        best_cer = current_cer  
        os.makedirs(str(epoch),exist_ok=True)
        model.save_pretrained("best/")

    os.makedirs("last/",exist_ok=True)
    model.save_pretrained("last/")

