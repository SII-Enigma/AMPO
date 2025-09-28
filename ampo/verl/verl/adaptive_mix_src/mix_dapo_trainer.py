# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import re
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Type, Dict, List, Optional
import random
import json

import torch.nn.functional as F

import numpy as np
import torch
from tqdm import tqdm

from omegaconf import OmegaConf, open_dict
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.utils.model import compute_position_id_with_mask

from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from tensordict import TensorDict


from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    # compute_advantage, 
    compute_response_mask,
    omega_conf_to_dataclass,
    agg_loss,
    should_save_ckpt_esi,
    StatefulDataLoader,
    ValidationGenerationsLogger,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.reward_score.repetition import detect_repetition_with_hash

import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, norm_adv_by_std_in_grpo=True, config: Optional[AlgoConfig] = None,):
    # prepare response group
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == 'gae':
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == 'grpo':
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == 'grpo_split':
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            on_policy_mask=on_policy_mask,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

def replace_left_and_right_str(text, left_str):
    while text.startswith(left_str):
        text = text[len(left_str):]
    while text.endswith(left_str):
        text = text[:-len(left_str)]
    return text

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    else:
        token_ids = prompt_token_ids[:non_pad_index[-1][0]+1].tolist()
    return token_ids

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._decode_cache = {}
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.chat_template = self.tokenizer.chat_template
        self.decode_hit = 0
        self.decode_miss = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return self.tokenizer(text, **kwargs)
        return_tensors = kwargs.pop("return_tensors", None)
        raw_inputs = self.tokenizer(text, **kwargs, padding=True)
        inputs = {key: value if key != "input_ids" and key != "attention_mask" else [] for key, value in
                  raw_inputs.items()}
        for i in range(len(text)):
            if return_tensors == "pt":
                inputs["input_ids"].append(torch.tensor(raw_inputs["input_ids"][i]))
                inputs["attention_mask"].append(torch.tensor(raw_inputs["attention_mask"][i]))
            else:
                inputs["input_ids"].append(raw_inputs["input_ids"][i])
                inputs["attention_mask"].append(raw_inputs["attention_mask"][i])
        if return_tensors == "pt":
            inputs["input_ids"] = torch.tensor(raw_inputs["input_ids"])
            inputs["attention_mask"] = torch.tensor(raw_inputs["attention_mask"])
        return inputs

    def decode(self, input_ids, **kwargs):
        # Convert input_ids to a tuple for hashing if it's a tensor
        if hasattr(input_ids, "tolist"):
            cache_key = tuple(input_ids.tolist())
        else:
            cache_key = tuple(input_ids)

        # Add kwargs to cache key to ensure different kwargs get different cache entries
        cache_key = (cache_key, tuple(sorted(kwargs.items())))

        if cache_key in self._decode_cache:
            self.decode_hit += 1
            return self._decode_cache[cache_key]

        self.decode_miss += 1
        result = self.tokenizer.decode(input_ids, **kwargs)
        self._decode_cache[cache_key] = result
        return result

    def batch_decode(self, input_ids, **kwargs):
        return [
            self.decode(input_ids[i], **kwargs) for i in range(len(input_ids))
        ]

    def release_cache(self):
        self._decode_cache = {}
        self.decode_hit = 0
        self.decode_miss = 0

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

class MIXRayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 off_policy_reward_fn=None,
                 val_reward_fn=None,
                 device_name=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.off_policy_reward_fn = off_policy_reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)
            
        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        else:
            logger.warning(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False
            
        self.reward_manager = self.config.reward_model.get("reward_manager", "naive")
        self._create_dataloader()

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        
        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(data_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         config=self.config.data,
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len)

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.adaptive_mix_src.rl_dataset_with_target import ResumableRandomSampler
            # print(f"We shuffle the training data...")
            # train_dataloader_generator = torch.Generator()
            # train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            # print(f"We do not shuffle the training data...")
            sampler = SequentialSampler(data_source=self.train_dataset)

        num_workers = self.config.data["dataloader_num_workers"]
    
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    
        if 'prob' in self.reward_manager:
            from torch.utils.data import ConcatDataset
            train_dataset_repeat = ConcatDataset([
                RLHFDatasetWithTarget(data_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         config=self.config.data,
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len)
                for _ in range(2)
            ])
            sampler_repeat = SequentialSampler(data_source=train_dataset_repeat)
            self.train_dataloader_repeat = StatefulDataLoader(dataset=train_dataset_repeat,
                                            batch_size=self.config.data.train_batch_size,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            collate_fn=collate_fn,
                                            sampler=sampler_repeat)
        
        self.val_dataset = RLHFDataset(data_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       config=self.config.data,)
        
        val_batch_size = len(self.val_dataset)
            
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
    
    def _dump_generations(self, sources, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "source": sources,
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")
        
    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")
            
            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
                
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        
        if not self.config.trainer.get('val_only', False):
            if 'prob' in self.reward_manager:
                promptgt2scoreA = self.compute_promptgt2scoreA(0)

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.
        
        self.teacher_source_tracker = defaultdict(int)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids", "tgt_input_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids", "tgt_input_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                    
                # gen_batch.non_tensor_batch = new_batch.non_tensor_batch
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    if 'prob' in self.reward_manager:
                        # Decode all input IDs in the batch at once
                        prompts = self.tokenizer.batch_decode(
                            new_batch.batch['input_ids'], 
                            skip_special_tokens=False
                        )
                        # prompts = [prompt.replace(self.tokenizer.pad_token, '') for prompt in prompts]
                        prompts = [replace_left_and_right_str(prompt, self.tokenizer.pad_token) for prompt in prompts]

                        # Extract ground truths for the entire batch
                        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in new_batch]

                        # Combine prompts and ground truths to create keys for lookup
                        prompt_gt_keys = [prompt + '\n\n\n' + gt for prompt, gt in zip(prompts, ground_truths)]

                        # Check if any prompt_gt_key is missing in promptgt2scoreA
                        if any(key not in promptgt2scoreA for key in prompt_gt_keys):
                            print("Skipping batch due to missing scoreA.")  # Log for robustness
                            continue

                        # Assign scoreA to each item in the batch
                        for i, key in enumerate(prompt_gt_keys):
                            new_batch[i].non_tensor_batch['reward_model']['scoreA'] = promptgt2scoreA[key]
                    
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                    if self.config.algorithm.adv_estimator == 'remax':
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                            
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            if 'naive' in self.reward_manager:
                                reward_result = self.reward_fn(new_batch, return_dict=True)
                                reward_tensor = reward_result["reward_tensor"]
                                format_reward_tensor = reward_result["format_reward_tensor"]
                                # reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                reward_extra_infos_dict = {}     
                            else:
                                print(f"Using cross entropy reward...")
                                ground_truth_list = [new_batch[i_].non_tensor_batch['reward_model']['ground_truth'] for i_ in range(len(new_batch))]
                                # ground_truth_list = [item for item in ground_truth_list for _ in range(self.config.actor_rollout_ref.rollout.n)]
                                new_batch_pr = self.construct_new_batch_optimized(new_batch, ground_truth_list)
                            
                                new_batch = new_batch.union(new_batch_pr)
                                with marked_timer('old_log_prob_pr', timing_raw):
                                    old_log_prob_pr = self.actor_rollout_wg.compute_log_prob_pr(new_batch)
                                    new_batch = new_batch.union(old_log_prob_pr)
                                if 'prob' in self.reward_manager:
                                    reward_result = self.reward_fn(new_batch, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]
                                    scoreA_tensor = reward_result["scoreA_tensor"]
                                    scoreB_tensor = reward_result["scoreB_tensor"]
                                    format_reward_tensor = reward_result["format_reward_tensor"]
                                    # reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                    reward_extra_infos_dict = {}
                                elif 'mix' in self.reward_manager:
                                    reward_result = self.reward_fn(new_batch, return_dict=True) # reward_tensor.shape: torch.Size([40, 1024])
                                    reward_tensor = reward_result["reward_tensor"]
                                    exact_tensor = reward_result["exact_tensor"]
                                    pr_scoreB_tensor = reward_result["pr_scoreB_tensor"]
                                    pr_scoreA_tensor = reward_result["pr_scoreA_tensor"]
                                    straightA_tensor = reward_result["straightA_tensor"]
                                    format_reward_tensor = reward_result["format_reward_tensor"]
                                    new_batch.batch['token_level_pr'] = reward_result["pr_reward_tensor"]
                                    new_batch.batch['token_level_vr'] = reward_result["vr_reward_tensor"]
                                    reward_extra_infos_dict = {}
                                else:
                                    logger.warning("Reward manager not found.")
                                    
                            if self.config.reward_model.get("repetition_penalty", False):
                                # Decode all responses in a batch
                                responses = self.tokenizer.batch_decode(new_batch.batch['responses'], skip_special_tokens=True)
                                repetition_penalty_list = []
                                for i_, response_i in enumerate(responses):
                                    # Apply repetition penalty
                                    non_zero_indices = reward_tensor[i_].nonzero(as_tuple=True)
                                    repetition_penalty = detect_repetition_with_hash(response_i, window_size=10, max_repetitions_limit=self.config.reward_model.get("repetition_penalty_max_repetitions_limit", 10))
                                    reward_tensor[i_][non_zero_indices] += repetition_penalty
                                    repetition_penalty_list.append(repetition_penalty)
                                repetition_penalty_rate = sum([1 for i in range(len(repetition_penalty_list)) if repetition_penalty_list[i] != 0]) / len(repetition_penalty_list)
                                metrics.update({"critic/repetition_penalty_rate": repetition_penalty_rate})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor
                        
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                        
                        # MIX off-policy rollout
                        new_gen_batch_output = new_batch.pop(batch_keys=['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'token_level_scores'])
                        new_gen_batch_output.non_tensor_batch = new_batch.non_tensor_batch
                        new_gen_batch_output = self.mix_off_policy_rollout(gen_batch, new_gen_batch_output)
                        
                        new_batch = new_batch.union(new_gen_batch_output)
                        
                        # log avg prefix ratio
                        if 'prefix_ratios' in new_batch.meta_info.keys():
                            metrics['batch/avg_prefix_ratio'] = float(np.mean(new_batch.meta_info['prefix_ratios']))
                        
                        # log teacher usage
                        if len(self.teacher_source_tracker) > 0:
                            total_injections = sum(self.teacher_source_tracker.values())
                            metrics['teacher_usage/total_inject_num'] = total_injections
                            for teacher_source, total_count in self.teacher_source_tracker.items():
                                metrics[f'teacher_usage/{teacher_source}'] = total_count / total_injections
                            
                        # log injection_stats
                        if 'injection_stats' in new_batch.meta_info.keys():
                            injection_stats = new_batch.meta_info['injection_stats']
                            dynamic_metrics = {
                                'injection/total_samples': injection_stats['total_samples'],
                                'injection/injected_samples': injection_stats['injected_samples'],
                                'injection/injection_rate': injection_stats['injection_rate'],
                                'injection/success_rate': 1.0 - injection_stats['injection_rate'],
                                'injection/teacher_dependency': injection_stats['injection_rate'], 
                            }
                            metrics.update(dynamic_metrics)
                            if 'avg_effective_targets' in injection_stats:
                                metrics['batch/avg_effective_targets'] = round(injection_stats['avg_effective_targets'], 2)
                            if 'max_effective_targets' in injection_stats:
                                metrics['batch/max_effective_targets'] = injection_stats['max_effective_targets']
                            if 'min_effective_targets' in injection_stats:
                                metrics['batch/min_effective_targets'] = injection_stats['min_effective_targets']
                            if 'prob_reward_mean' in injection_stats:
                                metrics['injection/prob_reward_mean'] = injection_stats['prob_reward_mean']
                            if 'prob_reward_max' in injection_stats:
                                metrics['injection/prob_reward_max'] = injection_stats['prob_reward_max']
                            if 'prob_reward_min' in injection_stats:
                                metrics['injection/prob_reward_min'] = injection_stats['prob_reward_min']
                            if 'prob_reward_std' in injection_stats:
                                metrics['injection/prob_reward_std'] = injection_stats['prob_reward_std']
                        
                        # log format_reward and pr_reward
                        format_reward = format_reward_tensor.sum(-1)
                        if 'prob' in self.reward_manager:
                            metrics.update({# reward
                                'critic/scoreB/mean':
                                    torch.mean(scoreB_tensor.sum(-1)).detach().item(),
                                'critic/scoreB/max':
                                    torch.max(scoreB_tensor.sum(-1)).detach().item(),
                                'critic/scoreB/min':
                                    torch.min(scoreB_tensor.sum(-1)).detach().item(),
                                'critic/scoreA/mean':
                                    torch.mean(scoreA_tensor.sum(-1)).detach().item(),
                                'critic/scoreA/max':
                                    torch.max(scoreA_tensor.sum(-1)).detach().item(),
                                'critic/scoreA/min':
                                    torch.min(scoreA_tensor.sum(-1)).detach().item(),
                                'critic/format_rewards/mean':
                                    torch.mean(format_reward).detach().item(),
                                'critic/format_rewards/max':
                                    torch.max(format_reward).detach().item(),
                                'critic/format_rewards/min':
                                    torch.min(format_reward).detach().item(),
                            })
                        elif 'mix' in self.reward_manager:
                            metrics.update({
                                "critic/vr_score/mean":
                                    torch.mean(exact_tensor.sum(-1).float()).detach().item(),
                                "critic/pr_scoreB/mean":
                                    torch.mean(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                                "critic/pr_scoreB/max":
                                    torch.max(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                                "critic/pr_scoreB/min":
                                    torch.min(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                                'critic/scoreA/mean':
                                    torch.mean(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                                'critic/scoreA/max':
                                    torch.max(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                                'critic/scoreA/min':
                                    torch.min(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                                'critic/all_correct_rate':
                                    torch.mean((straightA_tensor[:,0] == 1.).float().mean(-1)).detach().item(),
                                'critic/all_wrong_rate':
                                    torch.mean((straightA_tensor[:,0] == -1.).float().mean(-1)).detach().item(),
                            })
                        else:
                            metrics.update({# reward
                                'critic/format_rewards/mean':
                                    torch.mean(format_reward).detach().item(),
                                'critic/format_rewards/max':
                                    torch.max(format_reward).detach().item(),
                                'critic/format_rewards/min':
                                    torch.min(format_reward).detach().item(),
                            })

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = new_batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                        else:
                            fail_value = 0
                            success_value = 1
                        
                        format_value = self.config.reward_model.format_coefficient
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = new_batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile
                
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                
    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
    
    def mix_off_policy_rollout(self, gen_batch: DataProto, gen_batch_out: DataProto):   
        
        prompts = gen_batch_out.batch['prompts']
        responses = gen_batch_out.batch['responses']
        input_ids = gen_batch_out.batch['input_ids']
        attention_mask = gen_batch_out.batch['attention_mask']
        position_ids = gen_batch_out.batch['position_ids']
        reward_tensor = gen_batch_out.batch['token_level_scores']
        
        batch_size = responses.size(0)
        # print(f'batch_size: {batch_size}')
        
        prefix_mask = torch.zeros_like(responses, dtype=torch.bool)
        
        n = self.config.actor_rollout_ref.rollout.n
        n_prefix = self.config.actor_rollout_ref.rollout.n_prefix
        
        if 'tgt_input_ids' not in gen_batch.batch or n_prefix <= 0: 
            gen_batch_out.batch['prefix_mask'] = prefix_mask
            return gen_batch_out
        
        if n_prefix > n:
            logger.warning(f"n_prefix ({n_prefix}) > n ({n}), setting n_prefix <= n, will use n_prefix = n")
            n_prefix = n
        
        prefix_strategy = self.config.actor_rollout_ref.rollout.get('prefix_strategy', 'random')
        injection_strategy = self.config.actor_rollout_ref.rollout.get('injection_strategy', 'adaptive')  # adaptive/hybrid/always
        success_threshold = n_prefix / n 
        
        if injection_strategy not in ['adaptive', 'hybrid', 'always']:
            logger.info(f"Invalid injection strategy: {injection_strategy}, using 'never' instead")
            return gen_batch_out
        
        prefix_steps = self.config.actor_rollout_ref.rollout.get('prefix_steps', 300)
        prefix_linear_max_ratio = self.config.actor_rollout_ref.rollout.get('prefix_linear_max_ratio', 0.8)
        if prefix_strategy == 'linear':
            prefix_linear_max_ratio = self.config.actor_rollout_ref.rollout.get('prefix_linear_max_ratio', 0.8)
            pass
        elif prefix_strategy == 'linear_max':
            prefix_ratio_windows = [(0, i*prefix_linear_max_ratio/10) for i in range(10, 0, -1)]
            prefix_step_windows = [(i*prefix_steps/10, (i+1)*prefix_steps/10) for i in range(10)]
        elif prefix_strategy == 'linear_variance':
            # prefix_linear_max_ratio = self.config.actor_rollout_ref.rollout.get('prefix_linear_max_ratio', 0.8)
            prefix_linear_max_var = self.config.actor_rollout_ref.rollout.get('prefix_linear_max_var', 0.1)
        elif prefix_strategy == 'reverse_linear':
            # prefix_linear_max_ratio = self.config.actor_rollout_ref.rollout.get('prefix_linear_max_ratio', 0.8)
            prefix_ratio_windows = [(0, (i+1)*prefix_linear_max_ratio/10) for i in range(10)]
            prefix_step_windows = [(i*prefix_steps/10, (i+1)*prefix_steps/10) for i in range(10)]
        elif prefix_strategy == 'fixed':
            assert self.config.actor_rollout_ref.rollout.prefix_share_across_samples == False, "Fixed strategy could not work with prefix_share_across_samples=True ! "
            # prefix_fixed_num = self.config.actor_rollout_ref.rollout.get('prefix_fixed_num', 2)
            n_prefix = n_prefix if n_prefix != -1 else n
            if n_prefix <= 1:
                prefix_fix_ratios = [self.config.actor_rollout_ref.rollout.min_prefix_ratio]
            else:
                ratio_step = (self.config.actor_rollout_ref.rollout.max_prefix_ratio - self.config.actor_rollout_ref.rollout.min_prefix_ratio) / (n_prefix-1)
                prefix_fix_ratios = [self.config.actor_rollout_ref.rollout.min_prefix_ratio + i*ratio_step for i in range(n_prefix)]
        
        global_steps = gen_batch.meta_info['global_steps'] - 1 # we start from 1
        
        if len(gen_batch.batch['tgt_input_ids'].shape) == 3:# [batch_size, k, max_target_length]
            off_policy_targets = gen_batch.batch['tgt_input_ids']
            target_source_list = gen_batch_out.non_tensor_batch['target_source_list']
        elif len(gen_batch.batch['tgt_input_ids'].shape) == 2:  # [batch_size, max_target_length]
            off_policy_targets =  gen_batch.batch['tgt_input_ids'].unsqueeze(1)  # [batch_size, 1, max_target_length]
            target_source_list = [['DeepSeek-R1'] for _ in range(batch_size)]
        else:
            logger.error(f"Unexpected target shape: {gen_batch.batch['tgt_input_ids'].shape}")
        
        scores = [0.0] * batch_size
        success_mask = []
        injection_decisions = []
        prefix_ratios = [0.0] * batch_size
        
        effective_targets_per_sample = []
        
        orignal_batch_size = batch_size // n
        
        scoreA_list, scoreB_list, scoreB_minus_scoreA_list = [], [], []
        for i in range(orignal_batch_size):
            start_idx = i * n
            end_idx = (i + 1) * n
            
            sample_batch = gen_batch_out.slice(start_idx, end_idx)
            
            sample_target_sources = target_source_list[start_idx]
            
            extra_info = gen_batch_out[start_idx].non_tensor_batch.get('extra_info', {})
            num_teacher_models = extra_info.get('num_teacher_models', n_prefix)

            available_off_policy_nums = len(sample_target_sources)
            
            if num_teacher_models != available_off_policy_nums:
                logger.warning(f"num_teacher_models ({num_teacher_models}) != available_targets ({available_off_policy_nums}), using available_off_policy_nums instead")
                num_teacher_models = available_off_policy_nums
            
            reward_model = gen_batch_out[start_idx].non_tensor_batch.get('reward_model', {})
            ground_truth = reward_model.get('ground_truth', None)
            
            effective_k = min(num_teacher_models, n_prefix) 
            effective_targets_per_sample.append(effective_k)
            
            correct_positions = []
            incorrect_positions = []
            
            sample_ratios = [0.0] * n
            
            for i_ in range(len(sample_batch)):
                global_idx = start_idx + i_
                score = reward_tensor[global_idx].sum().item()
                scores[global_idx] = score
                
                if score > self.config.reward_model.format_coefficient + 0.5:
                    correct_positions.append(i_)
                else: 
                    incorrect_positions.append(i_)
            
            success_rate = len(correct_positions) / n
            # self.success_threshold = effective_k / n
            success_threshold = 1 / n
            sample_success = success_rate >= success_threshold
            success_mask.append(sample_success)

            logger.info(f"Sample {i}: {len(correct_positions)}/{n} correct, "
                    f"success_rate={success_rate:.3f}, threshold={success_threshold}")
            
            on_policy_scores = scores.copy()

            if injection_strategy == 'adaptive':
                needs_injection = not success_mask[i]
            elif injection_strategy == 'hybrid':
                if not success_mask[i]:
                    needs_injection = True
                else:
                    needs_injection = np.random.random() < 0.3
            elif injection_strategy == 'always':  # 'always'
                needs_injection = True
            else:
                logging.warning(f"Unknown injection strategy: {injection_strategy}, using 'never' instead")
                needs_injection = False
                            
            injection_decisions.append(needs_injection)
                        
            if not needs_injection:
                continue  

            import random
            num_incorrect = len(incorrect_positions)
            if num_incorrect > 0: 
                if not self.config.actor_rollout_ref.rollout.prefix_share_across_samples:
                    assert prefix_strategy != 'linear', "Linear strategy is not implemented with prefix_share_across_samples=False ! "
                    if effective_k == -1:
                        if prefix_strategy == 'random':
                            for pos in incorrect_positions:
                                sample_ratios[pos] = random.uniform(self.config.actor_rollout_ref.rollout.min_prefix_ratio, self.config.actor_rollout_ref.rollout.max_prefix_ratio)
                        elif prefix_strategy == 'reverse_linear' or prefix_strategy == 'linear_max':
                            w_idx = -1
                            for idx in range(len(prefix_step_windows)):
                                if global_steps >= prefix_step_windows[idx][0] and global_steps <= prefix_step_windows[idx][1]:
                                    w_idx = idx
                                    break
                            if w_idx == -1: 
                                w_idx = 0
                            for pos in incorrect_positions:
                                sample_ratios[pos] = random.uniform(prefix_ratio_windows[w_idx][0], prefix_ratio_windows[w_idx][1])
                        elif prefix_strategy == 'fixed':
                            for idx, pos in enumerate(incorrect_positions):
                                if idx < len(prefix_fix_ratios):
                                    sample_ratios[pos] = prefix_fix_ratios[idx]
                    else:                                    
                        max_selections = min(effective_k, num_incorrect)
                        if prefix_strategy == 'random':
                            selected_incorrect_positions = random.sample(incorrect_positions, max_selections)
                            for pos in selected_incorrect_positions:
                                sample_ratios[pos] = random.uniform(self.config.actor_rollout_ref.rollout.min_prefix_ratio, self.config.actor_rollout_ref.rollout.max_prefix_ratio)
                        elif prefix_strategy == 'reverse_linear' or self.config.actor_rollout_ref.rollout.prefix_strategy == 'linear_max':
                            selected_incorrect_positions = random.sample(incorrect_positions, max_selections)
                            w_idx = -1
                            for idx in range(len(prefix_step_windows)):
                                if global_steps >= prefix_step_windows[idx][0] and global_steps <= prefix_step_windows[idx][1]:
                                    w_idx = idx
                                    break
                            if w_idx == -1:
                                w_idx = 0
                            for pos in selected_incorrect_positions:
                                sample_ratios[pos] = random.uniform(prefix_ratio_windows[w_idx][0], prefix_ratio_windows[w_idx][1])
                        elif prefix_strategy == 'fixed':
                            selected_incorrect_positions = incorrect_positions[:max_selections]
                            for idx, pos in enumerate(selected_incorrect_positions):
                                if idx < len(prefix_fix_ratios):
                                    sample_ratios[pos] = prefix_fix_ratios[idx]
                        else:
                            raise NotImplementedError(f"Prefix strategy {prefix_strategy} is not implemented!")
                else:                                
                    max_selections = min(effective_k, num_incorrect)
                    if prefix_strategy == 'linear':
                        selected_incorrect_positions = incorrect_positions[:max_selections]
                        ratio = min((global_steps / prefix_steps), 1.0)
                        prefix_ratio_base = prefix_linear_max_ratio * (1-ratio)
                    else: # default, use random prefix ratio
                        prefix_ratio_base = None
                    prefix_ratio = prefix_ratio_base if prefix_ratio_base is not None else random.uniform(self.config.actor_rollout_ref.rollout.min_prefix_ratio, self.config.actor_rollout_ref.rollout.max_prefix_ratio)
                    if max_selections > 0:
                        selected_incorrect_positions = random.sample(incorrect_positions, max_selections)
                        for pos in selected_incorrect_positions:
                            sample_ratios[pos] = prefix_ratio
                    else:
                        logger.info(f"Prefix share across samples enabled! effective_k is 0, n is set to {n}")
                        sample_ratios = [prefix_ratio] * n
                                
                assert len(sample_ratios) == n
            else:
                logger.info("The on-policy responses of the sample is all right!!")
                            
            for j, ratio in enumerate(sample_ratios):
                prefix_ratios[start_idx + j] = ratio
            
            sample_off_policy_targets = off_policy_targets[start_idx][:num_teacher_models]
            
            sample_off_policy_prompts = gen_batch.slice(start_idx, start_idx+1).repeat(repeat_times=num_teacher_models, interleave=True)
            
            sample_off_policy_batch = self.construct_off_policy_gen_batch(response=sample_off_policy_targets, prompts=sample_off_policy_prompts, batch_size=num_teacher_models)
            sample_off_policy_batch.non_tensor_batch = gen_batch_out.slice(start_idx, start_idx+num_teacher_models).non_tensor_batch
            
            if self.off_policy_reward_fn is not None:
                if num_teacher_models > 1:
                    off_policy_ground_truth_list = [ground_truth] * num_teacher_models
                    # off_policy_ground_truth_list = ground_truth_list[:num_teacher_models]
                    sample_off_policy_batch_pr = self.construct_new_batch_optimized(sample_off_policy_batch, off_policy_ground_truth_list)
                    # print(f'sample_off_policy_batch_pr = {sample_off_policy_batch_pr}')
                    sample_off_policy_batch = sample_off_policy_batch.union(sample_off_policy_batch_pr)
                    
                    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
                    gpu_count = self.config.trainer.n_gpus_per_node
                    if num_teacher_models % gpu_count > 0:
                        padded_batch, pad_size = pad_dataproto_to_divisor(sample_off_policy_batch, gpu_count)
                        
                        off_policy_old_log_prob_pr = self.actor_rollout_wg.compute_log_prob_pr(padded_batch)

                        off_policy_old_log_prob_pr = unpad_dataproto(off_policy_old_log_prob_pr, pad_size)
                    else:
                        off_policy_old_log_prob_pr = self.actor_rollout_wg.compute_log_prob_pr(sample_off_policy_batch)
                    
                    sample_off_policy_batch = sample_off_policy_batch.union(off_policy_old_log_prob_pr)
                                
                    sample_off_policy_reward_results = self.off_policy_reward_fn(sample_off_policy_batch, return_dict=True)
                    sample_off_policy_reward_tensor = sample_off_policy_reward_results['reward_tensor']
                    scoreA_tensor = sample_off_policy_reward_results['scoreA_tensor']
                    scoreB_tensor = sample_off_policy_reward_results['scoreB_tensor']
                    for i_ in range(len(sample_off_policy_batch)):
                        scoreA_ = scoreA_tensor[i_].sum().item()
                        scoreB_ = scoreB_tensor[i_].sum().item()
                        scoreA_list.append(scoreA_)
                        scoreB_list.append(scoreB_)
                        scoreB_minus_scoreA_list.append(scoreB_ - scoreA_)
                                
                    assert sample_off_policy_reward_tensor.shape[0] == num_teacher_models, \
                        f"Reward tensor batch size {sample_off_policy_reward_tensor.shape[0]} != num_teacher_models {num_teacher_models}"
                    sample_off_policy_batch.batch['rm_scores'] = sample_off_policy_reward_tensor
                                
                    sample_off_policy_batch, sort_indices = self._sort_batch_by_scores(sample_off_policy_batch, score_key='rm_scores')

                    sample_target_sources = [sample_target_sources[idx.item()] for idx in sort_indices]
                else:
                    logger.info(f"off_policy responses num <= 1, no sorting needed")
            elif self.config.reward_model.off_policy_reward_manager == 'random':
                if num_teacher_models > 1:
                    sample_off_policy_batch, sort_indices = self._sort_batch_by_scores(sample_off_policy_batch, score_key='random')
                    sample_target_sources = [sample_target_sources[idx.item()] for idx in sort_indices]
                else:
                    logger.info(f"off_policy responses num <= 1, no sorting needed")
            else:
                sample_off_policy_batch = sample_off_policy_batch
                sample_target_sources = sample_target_sources
                        
            available_targets_batch = sample_off_policy_batch.slice(0, effective_k)
            available_target_sources = sample_target_sources[:effective_k]
                        
            if 'prob' in self.reward_manager: 
                if 'rm_scores' in available_targets_batch.batch:
                    available_targets_batch = available_targets_batch
                else:
                    available_targets_reward_results = self.reward_fn(available_targets_batch, return_dict=True)
                    available_targets_reward_tensor = available_targets_reward_results['reward_tensor']
                    available_targets_batch.batch['rm_scores'] = available_targets_reward_tensor
            elif 'mix' in self.reward_manager:
                if 'rm_scores' in available_targets_batch.batch:
                    available_targets_batch.batch.pop('rm_scores')
                available_targets_reward_results = self.reward_fn(available_targets_batch, return_dict=True)
                available_targets_reward_tensor = available_targets_reward_results['reward_tensor']
                available_targets_batch.batch['token_level_pr'] = available_targets_reward_results['pr_reward_tensor']
                available_targets_batch.batch['token_level_vr'] = available_targets_reward_results['vr_reward_tensor']
                available_targets_batch.batch['rm_scores'] = available_targets_reward_tensor       
            else:
                if 'rm_scores' in available_targets_batch.batch:
                    available_targets_batch.batch.pop('rm_scores')
                available_targets_reward_results = self.reward_fn(available_targets_batch, return_dict=True)
                available_targets_reward_tensor = available_targets_reward_results['reward_tensor']
                available_targets_batch.batch['rm_scores'] = available_targets_reward_tensor
                                                
            available_indices = list(range(len(available_targets_batch)))
            
            # print(f'available_indices: {available_indices}')
            for j in incorrect_positions:
                global_idx = start_idx + j
                prefix_ratio = sample_ratios[j]

                if prefix_ratio > 0.5 and len(available_indices) > 0:
                    target_list_idx = j % len(available_indices) if len(available_indices) > 1 else 0
                    target_idx = available_indices[target_list_idx]
                    target_batch_item = available_targets_batch[target_idx] 
                    available_indices.pop(target_list_idx)
                    
                    teacher_source = available_target_sources[target_idx]
                    self.teacher_source_tracker[teacher_source] += 1
                    
                    prompt_ids = target_batch_item.batch['prompts']
                    prompt_length = prompt_ids.shape[-1] #prompt length，shape[-1] is the last dimension length
                    valid_response_length = target_batch_item.batch['attention_mask'][prompt_length:].sum()
                                
                    prefix_mask[global_idx, :valid_response_length] = True
                    
                    old_score = on_policy_scores[global_idx]
                    off_policy_score = target_batch_item.batch['rm_scores'].sum().item()
                    
                    if off_policy_score < 1.0:
                        logger.info(f"Targeted injection: sample {i}, position {j}, this off-policy score: {off_policy_score} is too low, will set it to 1.0")
                        target_batch_item.batch['rm_scores'][valid_response_length - 1] = 1.0
                    
                    off_policy_score = target_batch_item.batch['rm_scores'].sum().item()
                    
                    sample_correctness = 'correct' if old_score > self.config.reward_model.format_coefficient + 0.5 else 'incorrect'
                    scores[global_idx] = off_policy_score
                    
                                        
                    logger.info(f"Targeted injection: sample {i}, position {j} ({sample_correctness}->off-policy), "
                                f"Using teacher source: {teacher_source} for sample {i}, position {j}, "
                                f"effective_k={effective_k}, prefix_ratio: {prefix_ratio:.3f}, "
                                f"old_score: {old_score:.2f}, new_score: {off_policy_score:.3f}")
                    
                    prompts[global_idx] = target_batch_item.batch['prompts']              
                    responses[global_idx] = target_batch_item.batch['responses']
                    input_ids[global_idx] = target_batch_item.batch['input_ids']
                    attention_mask[global_idx] = target_batch_item.batch['attention_mask']
                    position_ids[global_idx] = target_batch_item.batch['position_ids']
                    reward_tensor[global_idx] = target_batch_item.batch['rm_scores']
                                
                    # print(f'new response: {response[global_idx]}')
                    # print(f'new reward: {reward_tensor[global_idx]}')
                    
        logger.info(f"Injection decisions: {sum(injection_decisions)}/{orignal_batch_size} samples "
                    f"will receive off-policy guidance")
        
        # Construct output batch
        mixed_batch = TensorDict(
            {
                'prompts': prompts,
                'responses': responses,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'prefix_mask': prefix_mask,
                'token_level_scores': reward_tensor,
            },
            batch_size=batch_size)

        meta_info = {
            'prefix_ratios': prefix_ratios,
        }

        injection_stats = {
            'total_samples': len(injection_decisions),
            'injected_samples': sum(injection_decisions),
            'injection_rate': sum(injection_decisions) / len(injection_decisions),
            'injection_decisions': injection_decisions,
            'scores_precomputed': True, 
            'total_score_computations': len(scores), 
            'effective_targets_per_sample': effective_targets_per_sample,
            'avg_effective_targets': sum(effective_targets_per_sample) / len(effective_targets_per_sample),
            'max_effective_targets': max(effective_targets_per_sample),
            'min_effective_targets': min(effective_targets_per_sample),
        }
        if len(scoreB_minus_scoreA_list) > 0:
            injection_stats.update({
                'prob_reward_mean': sum(scoreB_minus_scoreA_list) / len(scoreB_minus_scoreA_list),
                'prob_reward_max': max(scoreB_minus_scoreA_list),
                'prob_reward_min': min(scoreB_minus_scoreA_list),
                'prob_reward_std': np.std(scoreB_minus_scoreA_list),
            })
        meta_info['injection_stats'] = injection_stats

        avg_prefix_ratio = sum(prefix_ratios) / len(prefix_ratios) if prefix_ratios else 0.0
               
        logger.info(f"Process responses with pre-computed scores: "
                f"{len(scores)} total scores, "
                f"injection_rate: {injection_stats['injection_rate']:.3f}, "
                f"avg_effective_targets={injection_stats['avg_effective_targets']:.1f}, "
                f"avg_prefix_ratio: {avg_prefix_ratio:.3f}")
                    
        return DataProto(batch=mixed_batch, meta_info=meta_info)
    
    def construct_off_policy_gen_batch(self, response, prompts: DataProto, batch_size: int):
        
        batch_size = batch_size
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs_right_pad(self.tokenizer.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
        
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        meta_info = prompts.meta_info

        eos_token_id = self.tokenizer.eos_token_id
        
        # process off_policy responses
        # add eos token id to the end of the target
        responses = [
            _pre_process_inputs_right_pad(self.tokenizer.pad_token_id, response[i]) for i in range(len(response))
        ]
        responses = [
                responses[i] + [self.tokenizer.eos_token_id,] if len(responses[i]) > 0 else responses[i]
                for i in range(len(responses))
            ]
        
        for i in range(len(responses)):
            response_length = len(responses[i])
            response_length = min(response_length, self.config.data.max_response_length)
            response[i, :response_length] = torch.tensor(responses[i][:response_length], dtype=response.dtype, device=response.device)

        # Pad sequences if needed
        if response.shape[1] < self.config.data.max_response_length:
            response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.config.response_length).to(
                idx.device)

        seq = torch.cat([idx, response], dim=-1)

        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Construct output batch
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _sort_batch_by_scores(self, batch: DataProto, score_key: str = 'rm_scores') -> DataProto:
        try:
            if score_key == 'random':
                if hasattr(batch, 'batch') and len(batch.batch) > 0:
                    first_key = next(iter(batch.batch.keys()))
                    batch_size = batch.batch[first_key].shape[0] if hasattr(batch.batch[first_key], 'shape') else len(batch.batch[first_key])
                else:
                    raise ValueError("Cannot determine batch size for random sorting")
            else:
                if score_key not in batch.batch:
                    raise ValueError(f"Score key '{score_key}' not found in batch")
                
                scores_tensor = batch.batch[score_key]  # shape: [batch_size, response_length]
                batch_size = scores_tensor.shape[0]

            if score_key == 'random':
                sorted_indices = torch.randperm(batch_size)
                logger.info(f"Random sorting batch: sorted_indices = {sorted_indices.tolist()}")
            else:
                total_scores = scores_tensor.sum(dim=1)  # shape: [batch_size]

                sorted_indices = torch.argsort(total_scores, descending=True)
                
                logger.info(f"Sorting batch by {score_key}: scores = {total_scores.tolist()}, "
                           f"sorted_indices = {sorted_indices.tolist()}")

            sorted_batch_dict = {}
            for key, tensor in batch.batch.items():
                if isinstance(tensor, torch.Tensor):
                    sorted_batch_dict[key] = tensor[sorted_indices]
                else:
                    sorted_batch_dict[key] = tensor
                    
            sorted_non_tensor_batch = batch.non_tensor_batch if hasattr(batch, 'non_tensor_batch') else {}

            sorted_meta_info = batch.meta_info if hasattr(batch, 'meta_info') else {}

            sorted_batch = DataProto(
                batch=TensorDict(sorted_batch_dict, batch_size=batch_size),
                non_tensor_batch=sorted_non_tensor_batch,
                meta_info=sorted_meta_info
            )

            if score_key != 'random' and score_key in sorted_batch.batch:
                new_total_scores = sorted_batch.batch[score_key].sum(dim=1)
                logger.debug(f"Sorted scores: {new_total_scores.tolist()}")

                for i in range(len(new_total_scores) - 1):
                    if new_total_scores[i] < new_total_scores[i + 1]:
                        logger.warning(f"Sorting verification failed at position {i}: "
                                     f"{new_total_scores[i].item()} < {new_total_scores[i + 1].item()}")
            elif score_key == 'random':
                logger.debug(f"Random sorting completed, no score verification needed")
            
            return sorted_batch, sorted_indices
            
        except Exception as e:
            logger.error(f"Error in batch sorting: {e}")
            logger.error(f"Batch keys: {list(batch.batch.keys()) if hasattr(batch, 'batch') else 'No batch'}")
            if hasattr(batch, 'batch') and score_key in batch.batch:
                logger.error(f"Score tensor shape: {batch.batch[score_key].shape}")
            identity_indices = torch.arange(len(batch.batch[score_key]))
            return batch, identity_indices
            
    def get_scoreA(self, data):
        batch_input_ids = data.batch['input_ids'] # [256, 512]
        pad_token_str = self.tokenizer.pad_token
        eos_token_str = self.tokenizer.eos_token
        max_prompt_length, max_response_length = self.config.data.max_prompt_length, self.config.data.max_response_length
        data_list = []
        prompt_str_list, ground_truth_list = [], []
        for i in range(len(batch_input_ids)):
            data_item = data[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            prompt_str = self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=False)
            new_text = prompt_str + ' ' + ground_truth + ' ' + eos_token_str
            # new_text_rmpad = new_text.replace(self.tokenizer.pad_token, '')
            new_text_rmpad = replace_left_and_right_str(new_text, self.tokenizer.pad_token)
            if not new_text_rmpad.endswith(eos_token_str): # For a base model, the eos_token_str is the same as pad_token_str
                new_text_rmpad += eos_token_str
            outputs = self.tokenizer(new_text_rmpad, return_tensors='pt', add_special_tokens=False)
            input_ids = outputs['input_ids']
            attention_mask = outputs['attention_mask']
            if '<｜Assistant｜><think>' in self.tokenizer.chat_template:
                sep_str = '<｜Assistant｜><think>' + '\n'
            elif '<｜Assistant｜>' in self.tokenizer.chat_template:
                sep_str = '<｜Assistant｜>' + '\n'
            elif 'assistant<|end_header_id|>' in self.tokenizer.chat_template: # llama 3.1 8b Instruct
                sep_str = 'assistant<|end_header_id|>\n' + '\n'
            elif '<start_of_turn>model' in self.tokenizer.chat_template:
                sep_str = '<start_of_turn>model' + '\n'
            else:
                sep_str = '<|im_start|>assistant' + '\n'
            pos = self.locate_substring_tokens(new_text_rmpad, sep_str, self.tokenizer)

            prompts = input_ids[:, :pos[-1] + 1]
            responses = input_ids[:, pos[-1] + 1:]

            pos_gt = self.locate_substring_tokens(new_text_rmpad, ground_truth, self.tokenizer, ignore_end_text=eos_token_str) # list
            # Note that if GT is empty, this will report errors.
            ground_truth_ids = input_ids[:, pos_gt[0]:pos_gt[-1] + 1]
            start = (pos_gt[0]) - (pos[-1] + 1)


            # Pad prompts and responses for future packing
            left_pad_tuple = (max_prompt_length- prompts.shape[-1], 0)
            right_pad_tuple = (0, max_response_length - responses.shape[-1])

            prompts = F.pad(prompts, left_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_length before collate_fn
            responses = F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_response_length before collate_fn

            input_ids = torch.cat([prompts, responses], dim=-1)

            # pad right first
            position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
            attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
            # then pad left
            attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
            position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

            ground_truth_mask = torch.zeros_like(responses)
            ground_truth_mask[:, start:start + ground_truth_ids.shape[-1]] = 1 # Suppose the response is <think> ABC </think> <answer> DEF </answer>. Then the mask is on " DEF ".



            row_dict = {
                'prompts': prompts[0],
                'responses': responses[0],
                'input_ids': input_ids[0],
                'attention_mask': attention_mask[0],
                'position_ids': position_ids[0],
                'ground_truth_mask': ground_truth_mask[0],
            }

            # prompt_str_list.append(prompt_str.replace(pad_token_str, ''))
            prompt_str_list.append(replace_left_and_right_str(prompt_str, pad_token_str))
            ground_truth_list.append(ground_truth)
            data_list.append(row_dict)

        data_new: DataProto = DataProto.from_single_dict(self.collate_fn(data_list))
        old_log_probs = self.actor_rollout_wg.compute_log_prob(data_new)['old_log_probs'].batch
        scoreAs_list = []
        old_log_probs_in_gt_list = []
        for i in range(len(batch_input_ids)):
            ground_truth_mask = data_new[i].batch['ground_truth_mask']
            old_log_prob = old_log_probs[i]

            old_log_probs_in_gt = old_log_prob[ground_truth_mask.bool()]
            if self.config.reward_model.get('compute_score_name', None) == 'mean_exp_log_softmax':
                scoreA = torch.mean(torch.exp(old_log_probs_in_gt)).item()
            # mean log probs
            elif self.config.reward_model.get('compute_score_name', None) == 'mean_log_softmax':
                scoreA = torch.mean(old_log_probs_in_gt).item()
            # product of probs
            elif self.config.reward_model.get('compute_score_name', None) == 'exp_sum_log_softmax':
                scoreA = torch.exp(torch.sum(old_log_probs_in_gt)).item()
            # root of the product of probs
            elif self.config.reward_model.get('compute_score_name', None) == 'exp_mean_log_softmax':
                scoreA = torch.exp(torch.mean(old_log_probs_in_gt)).item() 
            else:
                raise ValueError
            scoreAs_list.append(scoreA)
            old_log_probs_in_gt_list.append(old_log_prob[ground_truth_mask.bool()])

        return scoreAs_list, prompt_str_list, ground_truth_list, old_log_probs_in_gt_list
    
    def locate_substring_tokens(self, full_string, substring, tokenizer, ignore_end_text=None):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.

        Args:
            full_string (str): The full string to tokenize.
            substring (str): The substring to locate in the full string.
            tokenizer_name (str): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        encoding = tokenizer(full_string, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]  # List of (start, end) byte positions for each token

        # Find the byte-level start and end positions of the substring in the full string
        if ignore_end_text is not None:
            assert full_string.endswith(ignore_end_text), f"{full_string=} given but {ignore_end_text=} not in the end of the full string"
            sub_start = full_string[:-len(ignore_end_text)].rfind(substring)
        else:
            sub_start = full_string.rfind(substring)
        if sub_start == -1:
            print(f"{full_string=}")
            raise ValueError(f"Substring `{substring}` not found in the full string.")
        sub_end = sub_start + len(substring)

        # Locate the tokens that overlap with the substring's byte range
        matching_token_indices = [
            i for i, (start, end) in enumerate(offsets)
            if start < sub_end and end > sub_start
        ]

        return matching_token_indices
    
    def compute_promptgt2scoreA(self, epoch: int) -> None:
        """
        Processes and logs the distribution of scoreA for the given epoch.
        
        Args:
            epoch (int): The current epoch number.
        """
        # Check if probabilistic reward is enabled in the configuration
        # if 'prob' not in self.config.reward_model.reward_manager:
        #     return

        # Set the seed for reproducibility
        current_seed = self.config.data.get('seed', 1) if epoch == 0 else random.randint(0, 2**32 - 1)
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        promptgt2scoreA = {}

        scoreA_list = []


        total_train_samples = len(self.train_dataloader.dataset)

        train_dataloader = self.train_dataloader_repeat

        for idx, batch_dict in tqdm(enumerate(train_dataloader), total=len(self.train_dataloader) + 4):
            print(f"{idx=} {len(promptgt2scoreA)=} {len(scoreA_list)=}. The goal is {total_train_samples}")
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            scoreAs, prompt_strs, ground_truths, old_log_probs_in_gt_list = self.get_scoreA(batch)
            # Process each item in the batch

            for i in range(len(batch)):
                prompt = replace_left_and_right_str(self.tokenizer.decode(
                    batch.batch[i]['input_ids'], 
                    skip_special_tokens=False
                ), self.tokenizer.pad_token)
                ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth']
                key = prompt + '\n\n\n' + ground_truth
                if key not in promptgt2scoreA:
                    promptgt2scoreA[key] = scoreAs[i]
                    scoreA_list.append(scoreAs[i])
            if idx >= len(self.train_dataloader) + 4:
                break

        save_path = './logs/promptgt2scoreA.json'
        with open(save_path, 'w') as file:
            print(f"We dump to {save_path}")
            json.dump(promptgt2scoreA, file)
            # assert False

        # Reset the seed to ensure consistent data order for training
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        print(f"{len(promptgt2scoreA)=}")
        return promptgt2scoreA
    
    def replace_answer_with_gt_batch(self, tokenizer, gen_ids_batch, gen_response_batch, ground_truth_batch,
                                     prompts_batch_shape, start_think, end_think, eos_token_str,
                                     pad_token_str, start_answer, end_answer, max_length, suffix,
                                     other_answer=False):
        batch_size = len(gen_ids_batch)

        gen_texts = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=False)
        gen_response_texts = tokenizer.batch_decode(gen_response_batch, skip_special_tokens=False)

        gen_texts_rmpad = [replace_left_and_right_str(text, pad_token_str) for text in gen_texts]
        gen_response_texts_rmpad = [replace_left_and_right_str(text, pad_token_str) for text in gen_response_texts]

        for i in range(batch_size):
            if not gen_texts_rmpad[i].endswith(eos_token_str): # not in gen_texts_rmpad[i]:
                gen_texts_rmpad[i] += eos_token_str
            # if eos_token_str not in gen_response_texts_rmpad[i]:
            if not gen_response_texts_rmpad[i].endswith(eos_token_str):
                gen_response_texts_rmpad[i] += eos_token_str

        new_texts = []
        valid_flags = []

        for i in range(batch_size):
            gen_text_rmpad = gen_texts_rmpad[i]
            gen_response_text_rmpad = gen_response_texts_rmpad[i]
            ground_truth = ground_truth_batch[i]

            if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                start_think_count = gen_response_text_rmpad.count(start_think)
                end_think_count = gen_response_text_rmpad.count(end_think)
            middle_content, leading_whitespace, trailing_whitespace = ' ', ' ', ' '

            start_answer_tag = '<answer>'
            start_answer_count = gen_response_text_rmpad.count(start_answer_tag)
            if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                pattern = r'^.*' + start_think + r'.*' + end_think + r'.*' + start_answer_tag + r'.*$'
            elif self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1_nothink':
                pattern = r'^.*' + start_answer_tag + r'.*$'
            valid_flag = (
                    start_answer_count == 1 and
                    (re.fullmatch(pattern, gen_response_text_rmpad, re.DOTALL) is not None)
            )
            if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                valid_flag = (
                    valid_flag and 
                    start_think_count == 1 and
                    end_think_count == 1
                )

            if valid_flag:
                if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                    middle_content = gen_response_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
                answer_section = gen_response_text_rmpad.split(start_answer_tag)[1]

                if not answer_section.strip():
                    valid_flag = False
                else:
                    leading_whitespace = ''
                    for char in answer_section:
                        if char in [' ', '\n', '\t', '\r']:
                            leading_whitespace += char
                        else:
                            break
                    if self.config.reward_model.get("gt_tokens_one_more", False):
                        match = re.search('(\s*)</answer>', answer_section)
                        if match:
                            trailing_whitespace = match.group(1)

            if not self.config.reward_model.get("allow_empty_leading_whitespaces", False):
                leading_whitespace = leading_whitespace if leading_whitespace != '' else ' '
                leading_whitespace = '' if other_answer else leading_whitespace

            if self.config.reward_model.get("gt_tokens_one_more", False):
                pass
            else:
                trailing_whitespace = trailing_whitespace if trailing_whitespace != '' else ' '
                trailing_whitespace = '' if other_answer else trailing_whitespace

            if valid_flag:
                if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                    new_text = (end_think.join(gen_text_rmpad.split(end_think)[:-1]) +
                                end_think + middle_content + start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace +
                                end_answer + eos_token_str)
                elif self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1_nothink':
                    new_text = (start_answer.join(gen_text_rmpad.split(start_answer)[:-1]) + 
                                start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace + 
                                end_answer + eos_token_str)
            else:
                if self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1':
                    end_text = (end_think + middle_content + start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace +
                                end_answer + eos_token_str)
                elif self.config.reward_model.get('format_mode', 'R1_nothink') == 'R1_nothink':
                    end_text = (start_answer + 
                                leading_whitespace + ground_truth + trailing_whitespace + 
                                end_answer + eos_token_str)

                end_text_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(end_text))
                new_text = (replace_left_and_right_str(tokenizer.decode(gen_ids_batch[i][:-len(end_text_ids) - 5],
                                                  skip_special_tokens=False), tokenizer.pad_token) +
                            end_text)

            new_texts.append(new_text)
            valid_flags.append(valid_flag)


        batch_input_data = tokenizer(new_texts, return_tensors='pt',
                                          add_special_tokens=False, truncation=True,
                                          max_length=max_length)

        batch_input_ids = batch_input_data['input_ids']
        batch_attention_mask = batch_input_data['attention_mask']

        pos_startanswer_batch = self.batch_locate_substring_tokens(tokenizer, new_texts, start_answer)
        pos_gt_batch = self.batch_locate_substring_tokens(tokenizer, new_texts, ground_truth_batch,
                                                     ignore_end_text=end_answer + eos_token_str)

        if self.config.reward_model.get("gt_tokens_one_more", False):
            for pos_gt in pos_gt_batch:
                if pos_gt:
                    pos_gt.append(pos_gt[-1] + 1)

        batch_results = {
            f'prompts{suffix}': [],
            f'responses{suffix}': [],
            f'input_ids{suffix}': [],
            f'attention_mask{suffix}': [],
            f'position_ids{suffix}': [],
            f'ground_truth_mask{suffix}': [],
        }

        for i in range(batch_size):
            if not valid_flags[i] or batch_input_ids[i].shape[0] > max_length:
                valid_flags[i] = False

            pos_startanswer = pos_startanswer_batch[i]
            pos_gt = pos_gt_batch[i]

            if pos_startanswer and pos_gt and len(pos_startanswer) > 0 and len(pos_gt) > 0:
                start_pos = pos_startanswer[0]
                gt_start = pos_gt[0]
                gt_end = pos_gt[-1]

                prompts = batch_input_ids[i:i + 1, :start_pos]
                responses = batch_input_ids[i:i + 1, start_pos:]

                ground_truth_ids = batch_input_ids[i:i + 1, gt_start:gt_end + 1]
                start = gt_start - start_pos

                # Padding
                left_pad_tuple = (max_length - prompts.shape[-1], 0)
                
                if self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu - max_length > self.config.data.max_response_length:
                    right_pad_tuple = (0, self.config.data.max_response_length - responses.shape[-1])
                else:
                    right_pad_tuple = (0, self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu - max_length - responses.shape[-1])

                prompts = F.pad(prompts, left_pad_tuple, 'constant', tokenizer.pad_token_id)
                responses = F.pad(responses, right_pad_tuple, 'constant', tokenizer.pad_token_id)

                input_ids = torch.cat([prompts, responses], dim=-1)

                attention_mask = batch_attention_mask[i:i + 1]
                position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
                attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
                attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
                position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

                ground_truth_mask = torch.zeros_like(responses)
                if valid_flags[i]:
                    ground_truth_mask[:, start:start + ground_truth_ids.shape[-1]] = 1

            else:
                prompts = torch.zeros(1, max_length, dtype=torch.long)
                responses = torch.zeros(1, self.config.data.max_response_length, dtype=torch.long)
                input_ids = torch.cat([prompts, responses], dim=-1)
                attention_mask = torch.zeros_like(input_ids)
                position_ids = torch.zeros_like(input_ids)
                ground_truth_mask = torch.zeros_like(responses)

            batch_results[f'prompts{suffix}'].append(prompts[0])
            batch_results[f'responses{suffix}'].append(responses[0])
            batch_results[f'input_ids{suffix}'].append(input_ids[0])
            batch_results[f'attention_mask{suffix}'].append(attention_mask[0])
            batch_results[f'position_ids{suffix}'].append(position_ids[0])
            batch_results[f'ground_truth_mask{suffix}'].append(ground_truth_mask[0])

        for key in batch_results:
            batch_results[key] = torch.stack(batch_results[key])

        return batch_results
    
    def batch_locate_substring_tokens(self, tokenizer, full_strings, substrings, ignore_end_text=None):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.
        Args:
            full_string (List[str]): The full string to tokenize.
            substring (List[str]): The substring to locate in the full string.
            tokenizer_name (List[str]): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        batch_encodings = tokenizer(full_strings, return_offsets_mapping=True, add_special_tokens=False)
        batch_offsets = batch_encodings["offset_mapping"]  # List of (start, end) byte positions for each token
        # Find the byte-level start and end positions of the substring in the full string
        batch_matching_token_indices = []
        for string_idx in range(len(full_strings)):
            full_string = full_strings[string_idx]
            if isinstance(substrings, str):
                substring = substrings
            else:
                substring = substrings[string_idx]
            offsets = batch_offsets[string_idx]
            if ignore_end_text is not None:
                assert full_string.endswith(
                    ignore_end_text), f"{full_string=} given but {ignore_end_text=} not in the end of the full string"
                sub_start = full_string[:-len(ignore_end_text)].rfind(substring)
            else:
                sub_start = full_string.rfind(substring)
            if sub_start == -1:
                print(f"{full_string=}")
                raise ValueError(f"Substring `{substring}` not found in the full string.")
            sub_end = sub_start + len(substring)
            # Locate the tokens that overlap with the substring's byte range
            matching_token_indices = [
                i for i, (start, end) in enumerate(offsets)
                if start < sub_end and end > sub_start
            ]
            batch_matching_token_indices.append(matching_token_indices)
        return batch_matching_token_indices

    def construct_new_batch_optimized(self, gen_batch_output, ground_truth_list,
                                      start_think='<think>', end_think='</think>',
                                      start_answer='<answer>', end_answer='</answer>',
                                      suffix='_pr'):
        tokenizer = TokenizerWrapper(tokenizer=self.tokenizer)

        gen_ids = gen_batch_output.batch['input_ids']  # prompt + response
        gen_responses = gen_batch_output.batch['responses']  # response only

        pad_token_str = tokenizer.pad_token
        eos_token_str = tokenizer.eos_token
        max_length = self.config.data.max_prompt_length + self.config.data.max_response_length

        batch_results = self.replace_answer_with_gt_batch(
            tokenizer,
            gen_ids,
            gen_responses,
            ground_truth_list,
            gen_batch_output.batch['prompts'].shape[-1],
            start_think,
            end_think,
            eos_token_str,
            pad_token_str,
            start_answer,
            end_answer,
            max_length,
            suffix
        )

        gen_batch_output: DataProto = DataProto.from_single_dict(batch_results)
        # self.tokenizer = self.tokenizer.tokenizer
        return gen_batch_output

def compute_data_metrics_ours(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_on_example_ratio = off_policy_mask.sum().item() / on_policy_mask.sum().item()

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    # on/off prompt score
    # batch_size = batch.batch.batch_size[0] / n_samples
    # on_prompt_score, off_prompt_score = [], []
    # sequence_score = sequence_score.reshape(batch_size, n_samples, sequence_score.shape[-1]) # [bsz, n, l]
    # for i in range(batch_size):
    #     on_prompt_score.append(sequence_score[i][on_policy_mask[i]].mean())
    #     off_prompt_score.append(sequence_score[i][off_policy_mask[i]].mean())

    # on_prompt_score = torch.cat(on_prompt_score, dim=0)
    # off_prompt_score = torch.cat(off_prompt_score, dim=0)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        # 'on_off_metrics/on_prompt_score':
        #     torch.mean(on_prompt_score).detach().item(),
        # 'on_off_metrics/off_prompt_score':
        #     torch.mean(off_prompt_score).detach().item(),
        'on_off_metrics/off_on_example_ratio':
            off_on_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics
