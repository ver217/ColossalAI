from typing import Any, Callable, Dict, List, Optional
import torch
from torch.optim import Adam

from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models.base import Actor, Critic
from coati.models.generation_utils import update_model_kwargs_fn
from coati.models.loss import PolicyLoss, ValueLoss
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy, Strategy
from coati.trainer.callbacks import Callback

from colossalai.nn.optimizer import HybridAdam

import ray


from .utils import is_rank_0, get_actor_from_args, get_critic_from_args, get_strategy_from_args, set_dist_env, \
    state_dict_to

from .detached_trainer_base import DetachedTrainer


@ray.remote(concurrency_groups={"buffer_length": 1, "buffer_append": 1, "buffer_sample": 1, "model_io": 1, "compute": 1})
class DetachedPPOTrainer(DetachedTrainer):
    '''
        Detached Trainer for PPO algorithm
    Args:
        strategy (Strategy): the strategy to use for training
        model (str) : for actor / critic init
        pretrained (str) : for actor / critic init
        lora_rank (int) : for actor / critic init
        train_batch_size (int, defaults to 8): the batch size to use for training
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    '''

    def __init__(self,
                 experience_maker_holder_name_list: List[str],
                 strategy: str,
                 model: str,
                 pretrained: str = None,
                 lora_rank: int = 0,
                 rm_model: str = None,  # if not None, use below rm settings for critic
                 rm_pretrained: str = None,
                 rm_lora_rank: int = 0,
                 env_info: Dict[str, str] = None,
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.4,
                 experience_batch_size: int = 8,
                 max_epochs: int = 10,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        # set environment variables
        if env_info:
            set_dist_env(env_info=env_info)
        # configure strategy
        self.strategy = get_strategy_from_args(strategy)
        # configure models, loss and optimizers
        if rm_model is None:
            rm_model = model
            rm_pretrained = pretrained
            rm_lora_rank = lora_rank

        with self.strategy.model_init_context():
            self.actor = get_actor_from_args(model, pretrained, lora_rank)
            self.critic = get_critic_from_args(rm_model, rm_pretrained, rm_lora_rank)

        if strategy != 'colossalai_gemini':
            self.actor.to(torch.float16).to(torch.cuda.current_device())
            self.critic.to(torch.float16).to(torch.cuda.current_device())

        if strategy.startswith('colossalai'):
            self.actor_optim = HybridAdam(self.actor.parameters(), lr=1e-7)
            self.critic_optim = HybridAdam(self.critic.parameters(), lr=1e-7)
        else:
            self.actor_optim = Adam(self.actor.parameters(), lr=1e-7)
            self.critic_optim = Adam(self.critic.parameters(), lr=1e-7)

        (self.actor, self.actor_optim), (self.critic, self.critic_optim) = \
            self.strategy.prepare((self.actor, self.actor_optim), (self.critic, self.critic_optim))

        # configure trainer
        generate_kwargs = _set_default_generate_kwargs(self.strategy, generate_kwargs, self.actor)
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)

        super().__init__(experience_maker_holder_name_list,
                         train_batch_size=train_batch_size,
                         buffer_limit=buffer_limit,
                         buffer_cpu_offload=buffer_cpu_offload,
                         experience_batch_size=experience_batch_size,
                         max_epochs=max_epochs,
                         dataloader_pin_memory=dataloader_pin_memory,
                         callbacks=callbacks,
                         **generate_kwargs)

        # for remote maker initialization
        self._model_str = model
        self._rm_model_str = rm_model
        self._pretrained = pretrained
        self._rm_pretrained = rm_pretrained

    @ray.method(concurrency_group="model_io")
    def _update_remote_makers(self):
        # TODO: balance duties
        if is_rank_0():
            self.update_target_holder_list(self.target_holder_name_list)
            for target_holder in self.target_holder_list:
                with torch.no_grad():
                    ray.get(target_holder.update_experience_maker.remote(self._get_unwrapped_actor(), self._get_unwrapped_critic()))
                    
                    
            with torch.no_grad():
                # actor
                chunk_start = True
                chunk_end = False
                g = self._get_actor_state_dict_shard()
                state_dict_shard = next(g)
                while True:
                    try:
                        state_dict_shard_next = next(g)
                    except StopIteration:
                        chunk_end = True

                    for target_holder in self.target_holder_list:
                        target_holder.update_experience_maker.remote(
                            new_actor_state_dict = state_dict_shard,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end)
                    chunk_start = False
                    if chunk_end:
                        break
                    state_dict_shard = state_dict_shard_next
                    
                # critic
                chunk_start = True
                chunk_end = False
                g = self._get_critic_state_dict_shard()
                state_dict_shard = next(g)
                while True:
                    try:
                        state_dict_shard_next = next(g)
                    except StopIteration:
                        chunk_end = True

                    for target_holder in self.target_holder_list:
                        target_holder.update_experience_maker.remote(
                            new_critic_state_dict = state_dict_shard,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end)
                    chunk_start = False
                    if chunk_end:
                        break
                    state_dict_shard = state_dict_shard_next

    @ray.method(concurrency_group="model_io")
    def initialize_remote_makers(self):
        # TODO: balance duties
        if is_rank_0():
            self.update_target_holder_list(self.target_holder_name_list)

            with torch.no_grad():
                # actor / initial_model
                chunk_start = True
                chunk_end = False
                g = self._get_actor_state_dict_shard()

                state_dict_shard = next(g)
                while True:
                    try:
                        state_dict_shard_next = next(g)
                    except StopIteration:
                        chunk_end = True

                    for target_holder in self.target_holder_list:
                        target_holder.initialize_experience_maker.remote(
                            actor_model=self._model_str,
                            actor_pretrained=self._pretrained,
                            actor_state_dict=state_dict_shard,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end)
                    chunk_start = False
                    if chunk_end:
                        break
                    state_dict_shard = state_dict_shard_next

                # critic / reward_model
                chunk_start = True
                chunk_end = False
                g = self._get_critic_state_dict_shard()
                state_dict_shard = next(g)
                while True:
                    try:
                        state_dict_shard_next = next(g)
                    except StopIteration:
                        chunk_end = True

                    for target_holder in self.target_holder_list:
                        target_holder.initialize_experience_maker.remote(
                            critic_model=self._rm_model_str,
                            critic_pretrained=self._rm_pretrained,
                            critic_state_dict=state_dict_shard,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end)
                    chunk_start = False
                    if chunk_end:
                        break
                    state_dict_shard = state_dict_shard_next

    @ray.method(concurrency_group="compute")
    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        experience.to_device(torch.cuda.current_device())
        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)
        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim)
        self.actor_optim.zero_grad()

        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)

        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim)
        self.critic_optim.zero_grad()
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def strategy_save_actor(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.actor, path, only_rank0)

    def strategy_save_critic(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.critic, path, only_rank0)

    def strategy_save_actor_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.actor_optim, path, only_rank0)

    def strategy_save_critic_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.critic_optim, path, only_rank0)

    def _get_unwrapped_actor(self):
        if False:
            pass
        elif isinstance(self.strategy, ColossalAIStrategy):
            ret = Actor(self.strategy._unwrap_model(self.actor))
            return ret
        elif isinstance(self.strategy, DDPStrategy):
            return Actor(self.strategy._unwrap_actor(self.actor))
        elif isinstance(self.strategy, NaiveStrategy):
            return self.actor

    def _get_unwrapped_critic(self):
        if False:
            pass
        elif isinstance(self.strategy, ColossalAIStrategy):
            ret = self.strategy._unwrap_model(self.critic)
            return ret
        elif isinstance(self.strategy, DDPStrategy):
            return self.critic.module
        elif isinstance(self.strategy, NaiveStrategy):
            return self.critic

    def _get_actor_state_dict_shard(self, **config):
        for state_dict in self.strategy.get_model_state_dict_shard(self.actor, **config):
            yield state_dict_to(state_dict)

    def _get_critic_state_dict_shard(self, **config):
        for state_dict in self.strategy.get_model_state_dict_shard(self.critic, **config):
            yield state_dict_to(state_dict)


def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs
