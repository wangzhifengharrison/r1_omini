import os
import argparse
import pandas as pd
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def extract_tags(output, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = output.find(start_tag) + len(start_tag)
    end_idx = output.find(end_tag)
    return output[start_idx:end_idx].strip()


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument('--modal', type=str, default='video_audio', help='Modal type (video or video_audio)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing video paths')
    parser.add_argument('--instruct', type=str, required=True, help='Instruction for the model')

    args = parser.parse_args()

    # 读取CSV文件
    df = pd.read_csv(args.csv_path)

    # 初始化BERT分词器
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(args.model_path)

    # 创建新的列来存储预测结果
    df['predicted_label'] = ''
    df['reason'] = ''
    video_path_dfew = './DFEW_all/'

    # 循环处理每个视频
    for index, row in df.iterrows():
        video_path = video_path_dfew + str(row['video_name'])+'.mp4'
        print(video_path)

        # 处理视频输入
        video_tensor = processor['video'](video_path)

        # 根据modal类型决定是否处理音频
        if args.modal == 'video_audio' or args.modal == 'audio':
            audio = processor['audio'](video_path)[0]
        else:
            audio = None

        # 执行推理
        output = mm_infer(video_tensor, args.instruct, model=model, tokenizer=tokenizer, modal=args.modal,
                          question=args.instruct, bert_tokeni=bert_tokenizer, do_sample=False, audio=audio)

        # 提取<think>和<answer>标签中的内容
        predicted_label = extract_tags(output, 'answer')
        reason = extract_tags(output, 'think')

        # 更新DataFrame
        df.at[index, 'predicted_label'] = predicted_label
        df.at[index, 'reason'] = reason
        print(predicted_label)

    # 将结果保存回CSV文件
    df.to_csv(args.csv_path, index=False)

# python inference_dfew_test.py --modal video_audio --model_path ./R1-Omni-0.5B --csv_path ./DFEW_all_instruction/set_1_train_test.csv --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
# python train_dfew_emotion.py --modal video_audio --model_path ./R1-Omni-0.5B --train_csv ./DFEW_all_instruction/set_1_train_test.csv --val_csv ./DFEW_all_instruction/set_1_train_test.csv --output_dir ./emotion_model_output --batch_size 2 --learning_rate 2e-5 --num_epochs 5 --gradient_accumulation_steps 8 --fp16
if __name__ == "__main__":
    main()