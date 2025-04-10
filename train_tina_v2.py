# train_tina.py
import os
import argparse
import pandas as pd
import json  # 添加json模块
from humanomni_tina import model_init, mm_train
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer, AdamW
import torch
from humanomni.constants import DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN  # <-- ADD THIS

# Environment setup
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_conversation_data(data, file_path):
    """保存对话数据到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="HumanOmni Training Script")
    parser.add_argument('--modal', type=str, default='video_audio', help='Modal type')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=5)
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.train_csv)
    video_base_path = './DFEW_all/'

    # 初始化训练和验证数据列表
    train_data = []
    val_data = []

    # Initialize components
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    disable_torch_init()
    model, processor, tokenizer = model_init(args.model_path)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        epoch_train_data = []
        
        for index, row in df.iterrows():
            # ========== Video/Audio Input ==========
            video_path = os.path.join(video_base_path, f"{row['video_name']}.mp4")

            # Process video (shape: [T, C, H, W])
            video_tensor = processor['video'](video_path)

            # Process audio if needed
            audio_tensor = processor['audio'](video_path)[0] if 'audio' in args.modal else None

            # ========== Instruction & Target ==========
            # Example instruction format for emotion recognition
            if args.modal == 'video':
                modal_token = DEFAULT_VIDEO_TOKEN
            elif args.modal == 'video_audio':
                modal_token = DEFAULT_VIDEO_TOKEN + '\n' + DEFAULT_AUDIO_TOKEN
            elif args.modal == 'image':
                modal_token = DEFAULT_IMAGE_TOKEN
            else:
                raise ValueError(f"Unsupported modal: {args.modal}")

            instruct = (
                "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
            )

            # 创建完整的输入（包括模态标记）
            full_instruct = f"{modal_token}\n{instruct}"

            # Target should match the expected answer format
            target_answer = f"<think>{row['reason']}.</think><answer>{row['predicted_label']}</answer>"

            # 为BERT模型准备输入
            question = "<video>\n<audio>\n As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?"
            # question = "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
            question_input = bert_tokenizer([question], return_tensors='pt', padding=True, truncation=True,
                                            add_special_tokens=True)

            try:
                # ========== Training Step ==========
                metrics = mm_train(
                    image_or_video=video_tensor,
                    instruct=full_instruct,  # 传入完整指令，包括模态标记
                    targets=target_answer,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    audio=audio_tensor,
                    modal=args.modal,
                    bert_tokeni=bert_tokenizer,
                    question=question,  # 添加问题文本
                    grad_clip=1.0
                )

                total_loss += metrics['loss']
                print(f"Epoch {epoch + 1}, Sample {index + 1}, Loss: {metrics['loss']:.4f}")

                # 保存对话数据
                conversation_data = {
                    "video": f"{row['video_name']}.mp4",
                    "conversations": [
                        {
                            "from": "human",
                            "value": full_instruct
                        },
                        {
                            "from": "gpt",
                            "value": metrics.get('output', target_answer)  # 使用模型的实际输出，如果没有则使用目标答案
                        }
                    ]
                }
                epoch_train_data.append(conversation_data)

            except AssertionError as e:
                # 打印断言错误并继续下一个样本
                print(f"Skipping sample {index + 1} due to assertion error: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing sample {index + 1}: {str(e)}")
                continue

        # 保存每个epoch的训练数据
        train_data.extend(epoch_train_data)
        save_conversation_data(train_data, os.path.join(args.output_dir, f"train_data_epoch_{epoch + 1}.json"))
        
        # 保存验证数据（这里使用相同的格式，实际应用中可能需要单独处理验证集）
        val_data = epoch_train_data[:len(epoch_train_data)//5]  # 示例：取20%作为验证数据
        save_conversation_data(val_data, os.path.join(args.output_dir, f"val_data_epoch_{epoch + 1}.json"))

        # 计算平均损失（仅使用成功的样本）
        avg_loss = total_loss / len(df) if total_loss > 0 else float('inf')
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存模型检查点
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt"))


if __name__ == "__main__":
    main()

    # python train_tina.py   --modal video_audio   --model_path ./R1-Omni-0.5B   --train_csv ./DFEW_all_instruction/set_1_train_test_only_100_samples.csv   --output_dir ./trained_models   --batch_size 2   --learning_rate 2e-5   --num_epochs 5