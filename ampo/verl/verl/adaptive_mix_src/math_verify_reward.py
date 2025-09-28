"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from collections import defaultdict
from typing import List, Union

from verl.deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

from verl.deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

from math_verify import parse, verify

from verl import DataProto
import torch
import re
from verl.utils.reward_score import gsm8k, math

from verl.deepscaler.rewards.math_reward import deepscaler_reward_fn, THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from verl.adaptive_mix_src.reward_with_format import deepscaler_reward_fn_impl1
from typing import List, Union
from verl.workers.reward_manager.format import format_reward

def deepscaler_reward_fn_nothink(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    solution_str = f"{THOUGHT_DELIMITER_START}\n{THOUGHT_DELIMITER_END}\n{solution_str}"
    return deepscaler_reward_fn(solution_str, ground_truth, enable_llm)

def _select_rm_score_fn(data_source, reward_impl_version, phase='train'):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        if reward_impl_version == 0:
            return deepscaler_reward_fn
        elif reward_impl_version == 1:
            return deepscaler_reward_fn_impl1
        elif reward_impl_version == 2:
            return deepscaler_reward_fn_nothink
        elif reward_impl_version == 3:
            return reward_fn_math_verify
        elif reward_impl_version == 4:
            return reward_fn_math_verify_no_think
        elif reward_impl_version == 5:
            if phase == 'train':
                return reward_fn_prime_math_train
            elif phase == 'validation':
                return reward_fn_prime_math
            else:
                raise NotImplementedError(f"phase {phase} not in the implementation")
        else:
            raise NotImplementedError(f"reward_impl_version {reward_impl_version} not in the implementation, should be one of [0, 1, 2, 3, 4, 5]")

def extract_last_answer(text):
    # Find all occurrences of <answer>...</answer>
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if answers:
        return answers[-1]
    else:
        return ""

def remove_think_tags(input_text):
    # This pattern matches <think> followed by any content (non-greedy) until </think>
    pattern = r'<think>.*?</think>'
    # Substitute all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', input_text, flags=re.DOTALL)
    return cleaned_text
def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # print("think_format", self.config.think_format)
        if self.config.think_format:
            # Extract solution.
            if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
                model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
                # model_solution = remove_think_tags(model_response)
            else:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        else:
            model_solution = model_response
        # print(model_solution)
        # model_solution = model_response
        # model_solution = remove_think_tags(model_response)

        if '<answer>' in model_solution and '</answer>' in model_solution:
            model_solution = extract_last_answer(model_solution)
            
        labels = labeling_responses([model_solution,], input.ground_truth["answer"])
        if labels[0] is True:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def reward_fn_math_verify(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

def reward_fn_math_verify_no_think(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.think_format = False
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

def reward_fn_prime_math_train(solution_str: str, ground_truth: str):
    from verl.utils.reward_score.prime_math_train import compute_score
    is_correct, extracted_model_output = compute_score(solution_str, ground_truth)
    return is_correct

def reward_fn_prime_math(solution_str: str, ground_truth: str):
    from verl.utils.reward_score.prime_math import compute_score
    is_correct, extracted_model_output = compute_score(solution_str, ground_truth)
    return is_correct

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version, reward_fn_key="data_source", phase='train', format_coefficient=0, format_mode='R1_nothink') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_impl_version = reward_impl_version
        self.reward_fn_key = reward_fn_key
        self.phase = phase
        self.format_coefficient = format_coefficient
        self.format_mode = format_mode

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            format_score = format_reward(predict_str=response_str, format_mode=self.format_mode) 

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version, phase=self.phase)
            score = compute_score_fn(solution_str=response_str, ground_truth=ground_truth)
            
            if self.format_coefficient == -1:
                score = score if format_score == 1 else 0
            else:
                score = (1 - self.format_coefficient) * (score) + self.format_coefficient * format_score

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward
            format_reward_tensor[i, valid_response_length - 1] = format_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "format_reward_tensor": format_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor, format_reward_tensor