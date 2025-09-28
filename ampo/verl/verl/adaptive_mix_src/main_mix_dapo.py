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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from .mix_dapo_trainer import MIXRayDAPOTrainer
from .math_verify_reward import RewardManager


@hydra.main(config_path="config", config_name="mix_dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        # processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from .mix_fsdp_worker import MIXActorRolloutRefWorker, MIXAsyncActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(MIXActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        if config.data.num_off_policy_targets > 0 and 'prob' in config.reward_model.off_policy_reward_manager:
            from verl.workers.reward_manager import ProbRewardManager
            reward_manager_cls = ProbRewardManager
            off_policy_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, 
                compute_score_name=config.reward_model.get('compute_score_name', None), 
                shaping_function_name=config.reward_model.get('reward_manager_shaping_function_name', 'identity'),
                discrete_function_name=config.reward_model.get('reward_manager_discrete_function_name', 'identity'),
                format_coefficient=config.reward_model.get('format_coefficient', 0.1),
                reward_type=config.reward_model.get('reward_type', 'pr'),
                gt_tokens_one_more=config.reward_model.get('gt_tokens_one_more', False), 
                gt_tokens_one_more_adjusted=config.reward_model.get('gt_tokens_one_more_adjusted', False),
                format_mode=config.reward_model.get('format_mode', 'R1_nothink'),
            )
        else:
            off_policy_reward_fn = None
        
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
            
        if reward_manager_name == 'naive':
            reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                                      reward_impl_version=config.data.reward_impl_version,
                                      format_coefficient=config.reward_model.get('format_coefficient', 0.1),
                                      format_mode=config.reward_model.get('format_mode', 'R1_nothink'),
                                      )
            # from verl.workers.reward_manager import NaiveRewardManager
            # reward_manager_cls = NaiveRewardManager
            # reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=None, format_mode=config.reward_model.get('format_mode', 'R1'))
        elif 'mix' in reward_manager_name:
            from verl.workers.reward_manager import MixRewardManager
            reward_manager_cls = MixRewardManager
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer, num_examine=0,
                compute_exact_score_func=None,
                compute_fuzzy_score_name=config.reward_model.get('compute_score_name', None),
                shaping_function_name=config.reward_model.get('reward_manager_shaping_function_name', 'identity'),
                discrete_function_name=config.reward_model.get('reward_manager_discrete_function_name', 'identity'),
                n_rollouts=config.rollout.n,
                format_coefficient=config.reward_model.get('format_coefficient', 0.1),
                mix_type=config.reward_model.get('mix_type', 'hard'),
                pr_weight=config.reward_model.get('pr_weight', 0.5),
                vr_weight=config.reward_model.get('vr_weight', 1.0),
                format_mode=config.reward_model.get('format_mode', 'R1_nothink'),
            )
        elif 'prob' in reward_manager_name: # cross entropy
            from verl.workers.reward_manager import ProbRewardManager
            reward_manager_cls = ProbRewardManager
            reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, 
                compute_score_name=config.reward_model.get('compute_score_name', None), 
                shaping_function_name=config.reward_model.get('reward_manager_shaping_function_name', 'identity'),
                discrete_function_name=config.reward_model.get('reward_manager_discrete_function_name', 'identity'),
                format_coefficient=config.reward_model.get('format_coefficient', 0.1),
                reward_type=config.reward_model.get('reward_type', 'pr'),
                gt_tokens_one_more=config.reward_model.get('gt_tokens_one_more', False), 
                gt_tokens_one_more_adjusted=config.reward_model.get('gt_tokens_one_more_adjusted', False),
                format_mode=config.reward_model.get('format_mode', 'R1_nothink'),
            )

        else:
            print(f"{reward_manager_name=}")
            raise NotImplementedError

        # Note that we always use function-based RM for validation
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, 
                                      reward_impl_version=config.data.val_reward_impl_version, 
                                      phase='validation',
                                      format_coefficient=0,
                                      format_mode=config.reward_model.get('format_mode', 'R1_nothink'),)
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = MIXRayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            off_policy_reward_fn=off_policy_reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
