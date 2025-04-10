#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer, TrainingArguments
from humanomni import model_init
from humanomni.utils import disable_torch_init
from humanomni.humanomni_trainer import HumanOmniTrainer

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class DFEWEmotionDataset(Dataset):
    def __init__(self, csv_path, processor, instruct, tokenizer, bert_tokenizer, modal="video_audio",
                 video_dir="./DFEW_all/"):
        """
        Dataset for DFEW emotion recognition training

        Args:
            csv_path: Path to the CSV file with video names and emotion labels
            processor: HumanOmni processor for video and audio
            instruct: Instruction prompt for the model
            tokenizer: Model tokenizer
            bert_tokenizer: BERT tokenizer
            modal: "video", "audio", or "video_audio"
            video_dir: Directory containing video files
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.instruct = instruct
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.modal = modal
        self.video_dir = video_dir

        # Define emotion classes (adjust based on your DFEW dataset)
        self.emotion_classes = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_dir, f"{row['video_name']}.mp4")

        # Process video
        video_tensor = self.processor['video'](video_path)

        # Process audio if needed
        audio = None
        if self.modal == 'video_audio' or self.modal == 'audio':
            audio = self.processor['audio'](video_path)[0]

        # Get ground truth emotion (adjust column name if needed)
        emotion_label = row['emotion']

        # Create formatted instruction with expected output format
        formatted_instruction = self.instruct

        # Format the target output with thinking and answer tags
        # This assumes your dataset has both emotion labels and reasoning
        # Adjust as needed based on your dataset structure
        target_output = f"<think>I can see that the person in the video is displaying {emotion_label}. " \
                        f"Their facial expressions and body language clearly indicate {emotion_label}.</think> " \
                        f"<answer>{emotion_label}</answer>"

        # Tokenize inputs
        inputs = self.tokenizer(formatted_instruction, return_tensors="pt")
        targets = self.tokenizer(target_output, return_tensors="pt")

        # Create the training sample
        sample = {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": targets.input_ids[0],
            "video": video_tensor,
            "audio": audio
        }

        return sample


def collate_fn(batch):
    """Custom collate function to handle variable-length inputs and multimodal data"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    videos = [item["video"] for item in batch]
    audios = [item["audio"] for item in batch if item["audio"] is not None]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    # Stack videos and audios if available
    videos = torch.stack(videos)
    if audios:
        audios = torch.stack(audios)
    else:
        audios = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "video": videos,
        "audio": audios
    }


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Training Script for Emotion Recognition")
    parser.add_argument('--modal', type=str, default='video_audio', help='Modal type (video, audio, or video_audio)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--instruct', type=str, default="As an emotional recognition expert; throughout the video, "
                                                        "which emotion conveyed by the characters is the most obvious to you? Output the thinking "
                                                        "process in <think> </think> and final emotion in <answer> </answer> tags.",
                        help='Instruction for the model')
    parser.add_argument('--output_dir', type=str, default='./emotion_model_output',
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--video_dir', type=str, default='./DFEW_all/', help='Directory containing video files')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')

    args = parser.parse_args()

    # Initialize BERT tokenizer
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Disable Torch initialization
    disable_torch_init()

    # Initialize model, processor, and tokenizer
    model, processor, tokenizer = model_init(args.model_path)

    # Create training dataset
    train_dataset = DFEWEmotionDataset(
        csv_path=args.train_csv,
        processor=processor,
        instruct=args.instruct,
        tokenizer=tokenizer,
        bert_tokenizer=bert_tokenizer,
        modal=args.modal,
        video_dir=args.video_dir
    )

    # Create validation dataset
    val_dataset = DFEWEmotionDataset(
        csv_path=args.val_csv,
        processor=processor,
        instruct=args.instruct,
        tokenizer=tokenizer,
        bert_tokenizer=bert_tokenizer,
        modal=args.modal,
        video_dir=args.video_dir
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="tensorboard",
        remove_unused_columns=False,  # Important for custom datasets
        dataloader_pin_memory=False,  # Important when using large videos
    )

    # Initialize trainer
    trainer = HumanOmniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    print(f"Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
# python train_dfew_emotion.py --modal video_audio --model_path ./R1-Omni-0.5B --train_csv ./DFEW_all_instruction/set_1_train_test.csv --val_csv ./DFEW_all_instruction/set_1_train_test.csv --output_dir ./emotion_model_output --batch_size 2 --learning_rate 2e-5 --num_epochs 5 --gradient_accumulation_steps 8 --fp16