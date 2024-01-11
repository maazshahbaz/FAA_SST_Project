#Requirements %%
#!pip install datasets>=2.6.1
#!pip install git+https://github.com/huggingface/transformers
#!pip install librosa
#!pip install evaluate>=0.30
#!pip install jiwer
#!pip install gradio
#!pip install -q bitsandbytes datasets accelerate
#!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main
#!pip install ipywidgets
#!pip install wandb

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
import wandb
wandb.login()
# Select CUDA device index
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_PROJECT"] = "LoRA_CD_R128"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
model_name_or_path = "openai/whisper-medium.en"
language = "english"
language_abbr = "en"
task = "transcribe"


# %%
import torch
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# %%
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split

dataset1 = load_dataset("Jzuluaga/uwb_atcc")
dataset2 = load_dataset("Jzuluaga/atcosim_corpus")
dataset3 = load_dataset("Jzuluaga/atco2_corpus_1h")

#%%
'''
# Concatenate all the datasets
all_datasets = concatenate_datasets([dataset1["train"], dataset1["test"], dataset2["train"], dataset2["test"], dataset3["test"]])

# Split the combined dataset into training and testing sets (1001 samples for testing)
dataset = all_datasets.train_test_split(test_size=0.0408)

# The 'dataset' object is now a DatasetDict with 'train' and 'test' splits
print(dataset)
'''
#%%

train = concatenate_datasets([dataset1["train"], dataset2["train"]])
test = concatenate_datasets([dataset3["test"]])

dataset=DatasetDict()

dataset['train']=train
dataset['test']=test

print(dataset)
# Remove unnecessary columns
dataset.remove_columns(["id", "segment_start_time", "segment_end_time", "duration"])

# %%
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

# %%
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

# %%
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# %%
from datasets import Audio

common_voice = dataset.cast_column("audio", Audio(sampling_rate=16000))

def add_noise(audio_array, noise_level=0.005):
    # Generate random noise with the same length as the input audio
    noise = np.random.normal(0, noise_level, len(audio_array))
    
    # Add the generated noise to the input audio
    audio_with_noise = audio_array + noise

    return audio_with_noise

# %%
def prepare_dataset(batch):
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    
    # Add noise to audio
    audio_array_with_noise = add_noise(audio["array"])

    # Compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio_array_with_noise, sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch

# %%
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)


# %%
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# %%
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %%
import evaluate

metric = evaluate.load("wer")

# %%
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# %%
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True)

# %%
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# %%

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# %%
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=128, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

# %%
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    report_to="wandb",
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    predict_with_generate = True,
)

# %%

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# %%
trainer.train()