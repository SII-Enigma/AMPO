#export HF_ENDPOINT=https://hf-mirror.com  
import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from math_verify import parse, verify
import re

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

SYSTEM_PROMPT = "You are a helpful assistant. Please reason step by step to solve the problem and put the final answer within the <answer> </answer> tags."

def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")
            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        return decorator

@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_prime_sft(question, tokenizer):
    # for math problem
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end." 
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_llama_math_template(question: str):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + question + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )

def extract_last_answer(text):
    # Find all occurrences of <answer>...</answer>
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if answers:
        return answers[-1]
    else:
        return ""
    
def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False):

    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(torch.cuda.device_count())
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9) 

    if force_generate or (not os.path.exists(dec_output_path)):
        messages = df['prompt'].tolist()
        
        assert remove_system is True
        if remove_system:
            print('remove system')
            assert messages[0][0]['role'] == 'system'
            messages = [message[1:] for message in messages]
            
        else:
            assert remove_system is False
            print('not remove system')
            
        answers = df['reward_model'].tolist()
        answers = [answer['ground_truth'] for answer in answers]
        # if debug:
            # answers = answers[:10]
        assert len(messages) == len(answers)
                
        print(messages[0])
        print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}")
        outputs, gen_prompts = generate_vllm(messages, llm, tokenizer, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        print('Example input: ', gen_prompts[0])
        # rets = {}
        
        # save the outputs first
        with open(dec_output_path, 'w') as fo:
            for i, output in enumerate(outputs):
                prompt = output.prompt
                for j in range(n):
                    generated_text = output.outputs[j].text
                    item = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'answer': answers[i]
                    }
                    fo.write(json.dumps(item) + '\n')
                    
        # format sort prompts, outputs, answers
        assert len(outputs[0].outputs) == n
        prompts = [out.prompt for out in outputs for j in range(n)]
        answers = [answers[i] for i in range(len(outputs)) for j in range(n)]
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]
        
        jss = []
        with open(dec_output_path, 'r') as f:
            for line in f:
                jss.append(json.loads(line))
        
        outputs = [item['generated_text'] for item in jss]
        prompts = [item['prompt'] for item in jss]
        answers = [item['answer'] for item in jss]
    
    data_sources = df['data_source'].tolist()
    
    from collections import defaultdict
    rets = defaultdict(list)
    response_lengths = defaultdict(list) 
    save_data = []
    avg = 0
    from tqdm import tqdm

    print('Scoring...')
    if skip_scoring:
        return
    
    # for i, output in tqdm(enumerate(outputs)):
    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        # print(i)
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        
        response_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        response_length = len(response_tokens)
        
        think_format = False
        answer_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            think_format = True
        if no_split_think:
            think_format = False

        if '<answer>' in generated_text and '</answer>' in generated_text:
            answer_format = True
            pattern_answer = extract_last_answer(generated_text)
        
        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                labels = [False]
        if answer_format:
            try:
                # generated_text = generated_text.split("<answer>")[0]
                generated_text = generated_text + f'The final answer is \\boxed{{{pattern_answer}}}.'
            except Exception as e:
                labels = [False]
                
        if labels is None:
            try:
                labels = labeling_responses([generated_text,], answer)
            except Exception as e:
                labels = [False]
        
        rets[data_sources[i]].append(labels[0])
        response_lengths[data_sources[i]].append(response_length)
        save_data.append({
            'prompt': prompt,
            'data_source': data_sources[i],
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0],
            'response_tokens': response_length
        })
        if labels[0]:
            avg += 1

    
    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    all_response_lengths = [length for lengths in response_lengths.values() for length in lengths]
    overall_avg_length = np.array(all_response_lengths).mean() if all_response_lengths else 0
    print(f'overall avg response tokens: {overall_avg_length:.2f}')
    
    accs = []
    avg_lengths = []
    evaluation_summary = {
        'overall_accuracy': avg / len(outputs),
        'overall_avg_response_tokens': overall_avg_length,
        'data_source_results': {}
    }
    
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        avg_length = np.array(response_lengths[data_source]).mean()
        print(f'{data_source}: accuracy={acc:.4f}, avg_response_tokens={avg_length:.2f}')
        accs.append(acc)
        avg_lengths.append(avg_length)
        
        evaluation_summary['data_source_results'][data_source] = {
            'accuracy': float(acc),
            'avg_response_tokens': float(avg_length),
            'sample_count': len(labels)
        }
    
    evaluation_summary['overall_avg_accuracy_by_source'] = float(np.array(accs).mean())
    evaluation_summary['overall_avg_tokens_by_source'] = float(np.array(avg_lengths).mean())
    
    print('avg acc by source: ', np.array(accs).mean())
    print('avg response tokens by source: ', np.array(avg_lengths).mean())
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        print(f'Results saved to: {output_file}')
        print(f'Summary saved to: {summary_file}')
        
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')

def generate_vllm(messages, llm, tokenizer, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1):
    gen_prompts = []
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=8192, n=n)
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'luffy':
            message = [
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[0]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'llama':
            message = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": cur_message[0]['content']}
            ]
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(cur_message[0]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs, gen_prompts

if __name__ == "__main__":
    import fire
    fire.Fire(main)
