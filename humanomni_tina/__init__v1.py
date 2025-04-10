import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, process_audio, tokenizer_multimodal_token, get_model_name_from_path, \
    KeywordsStoppingCriteria, process_image_npary
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
import transformers


def model_init(model_path=None, **kwargs):
    # with_face = kwargs.get('with_face', False)
    model_path = "HumanOmni_7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len, audio_processor = load_pretrained_model(model_path, None, model_name,
                                                                                      **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    if "qwen2vit" in model_path:
        from .mm_utils import process_image_qwen, process_video_qwen
        processor = {
            'image': partial(process_image_qwen, processor=processor, aspect_ratio=None),
            'video': partial(process_video_qwen, processor=processor, aspect_ratio=None, num_frames=num_frames),
        }
    else:
        processor = {
            'image': partial(process_image, processor=processor, aspect_ratio=None),
            'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
            'face': partial(process_image_npary, processor=processor, aspect_ratio=None),
            'audio': partial(process_audio, processor=audio_processor),
        }
    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None,
             **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,
                                      add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' + DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
                """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                """\n"""
                """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
             }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>",
                                                                                "<im_start><image><im_end>").replace(
            "<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(
        0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts.
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def mm_train(image_or_video, instruct, targets, model, tokenizer, optimizer, audio=None, modal='video', question=None, 
             bert_tokeni=None, **kwargs):
    """训练API，用于HumanOmni的多模态训练。

    Args:
        model: HumanOmni模型。
        image_or_video (torch.Tensor): 图像张量 (1, C, H, W) / 视频张量 (T, C, H, W)。
        instruct (str): 理解视频/图像的文本指令。
        targets (str): 目标输出文本。
        tokenizer: 分词器。
        optimizer: 优化器。
        audio (torch.Tensor, optional): 音频张量。
        modal (str): 训练模态。
        question (str, optional): 问题文本。
        bert_tokeni: BERT分词器（如适用）。
    Returns:
        dict: 包含损失和其他训练指标的字典。
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,
                                     add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' + DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"不支持的模态: {modal}")

    # 1. 视觉预处理（加载并转换图像或视频）
    if modal == 'text' or modal == 'audio':
        tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    if audio is not None:
        audio = audio.half().cuda()

    # 2. 文本预处理（标签处理和生成提示）
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"不支持的指令类型: {type(instruct)}")

    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
                """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                """\n"""
                """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
             }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # 添加模态包装标记
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>",
                                                                               "<im_start><image><im_end>").replace(
            "<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(
        0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 准备目标输出
    if isinstance(targets, str):
        target_message = [{'role': 'assistant', 'content': targets}]
        target_text = tokenizer.apply_chat_template(target_message, tokenize=False, add_generation_prompt=False)
        target_ids = tokenizer(target_text, return_tensors='pt').input_ids.cuda()
    else:
        target_ids = targets.cuda()

    # 3. 训练步骤
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_masks,
        images=tensor,
        labels=target_ids,
        prompts=question_prompt,
        audios=audio
    )

    loss = outputs.loss
    
    # 反向传播和优化
    loss.backward()
    
    # 梯度裁剪（可选）
    grad_clip = kwargs.get('grad_clip', 1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    # 返回训练指标
    metrics = {
        'loss': loss.item(),
    }
    
    if hasattr(outputs, 'logits'):
        with torch.no_grad():
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            
            # 检查并修复形状不匹配问题
            if pred_ids.shape != target_ids.shape:
                print(f"警告: 预测ID形状与目标ID形状不匹配!")
                print(f"预测ID形状: {pred_ids.shape}, 目标ID形状: {target_ids.shape}")
                
                # 计算可以比较的最小公共长度
                min_batch_size = min(pred_ids.shape[0], target_ids.shape[0])
                min_seq_len = min(pred_ids.shape[1] if len(pred_ids.shape) > 1 else pred_ids.shape[0], 
                                target_ids.shape[1] if len(target_ids.shape) > 1 else target_ids.shape[0])
                
                # 根据维度数调整比较方式
                if len(pred_ids.shape) > 1 and len(target_ids.shape) > 1:
                    # 两者都是2D张量
                    comparison = (pred_ids[:min_batch_size, :min_seq_len] == target_ids[:min_batch_size, :min_seq_len])
                elif len(pred_ids.shape) > 1:
                    # pred_ids是2D，target_ids是1D
                    comparison = (pred_ids[0, :min_seq_len] == target_ids[:min_seq_len])
                elif len(target_ids.shape) > 1:
                    # pred_ids是1D，target_ids是2D
                    comparison = (pred_ids[:min_seq_len] == target_ids[0, :min_seq_len])
                else:
                    # 两者都是1D张量
                    comparison = (pred_ids[:min_seq_len] == target_ids[:min_seq_len])
                
                correct = comparison.sum().item()
                total = comparison.numel()
                print(f"比较了{total}个token，共{correct}个正确")
            else:
                # 形状匹配，直接计算
                correct = (pred_ids == target_ids).sum().item()
                total = target_ids.numel()
            
            accuracy = correct / total if total > 0 else 0
            metrics['accuracy'] = accuracy
    
    return metrics

