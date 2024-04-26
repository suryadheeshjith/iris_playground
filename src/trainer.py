from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.world_model import WorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed

import logging

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        name = cfg.name + "//" + cfg.sub_name if hasattr(cfg, "sub_name") else ".LOCAL" + "//" + cfg.name
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            name = name,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        self.is_ram = cfg.env.train._target_ == "envs.make_atari_ram"

        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.output_dir / 'checkpoints' if not cfg.bc.should else self.output_dir / 'bc_checkpoints'
        self.media_dir = self.output_dir / 'media'
        self.episode_dir = self.output_dir / 'episodes'
        self.reconstructions_dir = self.output_dir / 'reconstructions'

        if not cfg.common.resume:
            recopy = '.LOCAL' in cfg.output_dir
            self.ckpt_dir.mkdir(exist_ok=recopy, parents=False)
            self.media_dir.mkdir(exist_ok=recopy, parents=False)
            self.episode_dir.mkdir(exist_ok=recopy, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=recopy, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.bc.should:
            self.train_bc_dataset = instantiate(cfg.datasets.bc, train=True, is_ram=self.is_ram, main_folder=cfg.bc_datapath)
            self.test_bc_dataset = instantiate(cfg.datasets.bc, train=False, is_ram=self.is_ram, main_folder=cfg.bc_datapath)
            self.bc_env = create_env(cfg.env.test, 1)
        
        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train, self.is_ram)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test, self.is_ram)
        
        assert self.cfg.bc.should or (self.cfg.training.should or self.cfg.evaluation.should)

        env = train_env if self.cfg.training.should else test_env



        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        actor_critic = instantiate(cfg.actor_critic, act_vocab_size=env.num_actions, is_ram=self.is_ram)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        logging.info(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        logging.info(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        logging.info(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        
        if self.cfg.bc.should: 
            self.bc_optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.bc.learning_rate)
        else:
            self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        if cfg.training.load_bc_agent or cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)
            print("Successfully loaded model weights!")

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:

        if self.cfg.bc.should: 
            for epoch in range(self.cfg.bc.epochs):
                logging.info(f"\nEpoch {epoch} / {self.cfg.bc.epochs}\n")
                start_time = time.time()
                to_log = []

                to_log += self.bc_train_actor_critic(epoch)
                self.agent.actor_critic.eval()
                to_log += self.bc_eval_actor_critic(epoch)
                to_log += self.bc_eval_trajectory(epoch)

                self.save_checkpoint(epoch, save_agent_only=True)
                
                to_log.append({'duration': (time.time() - start_time) / 3600})
                for metrics in to_log:
                    wandb.log({'epoch': epoch, **metrics})
                    print({'epoch': epoch, **metrics})
        else:              
          for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
              logging.info(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
              start_time = time.time()
              to_log = []

              if self.cfg.training.should:
                  if epoch <= self.cfg.collection.train.stop_after_epochs:
                      to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                  to_log += self.train_agent(epoch)

              if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                  self.test_dataset.clear()
                  to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                  to_log += self.eval_agent(epoch)

              if self.cfg.training.should:
                  self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

              to_log.append({'duration': (time.time() - start_time) / 3600})
              for metrics in to_log:
                  wandb.log({'epoch': epoch, **metrics})
                  print({'epoch': epoch, **metrics})

        self.finish()

    def bc_train_actor_critic(self, epoch: int):
        self.agent.train()
        self.agent.zero_grad()

        metrics_actor_critic = {}

        cfg_actor_critic = self.cfg.bc.actor_critic
        component = self.agent.actor_critic
        steps_per_epoch = cfg_actor_critic.steps_per_epoch
        optimizer = self.bc_optimizer_actor_critic
        grad_acc_steps = cfg_actor_critic.grad_acc_steps
        batch_num_samples = cfg_actor_critic.batch_num_samples
        sequence_length = cfg_actor_critic.burn_in + 1
        max_grad_norm = cfg_actor_critic.max_grad_norm

        loss_total_epoch = 0.0
        accuracy_total_epoch = 0.0

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_bc_dataset.sample_batch(batch_num_samples)
                assert batch['observations'].shape[1] == sequence_length
                batch = self._to_device(batch)

                loss, acc = component.compute_bc_loss(batch)
                loss = loss / grad_acc_steps
                loss.backward()
                loss_total_epoch += loss.item() / steps_per_epoch
                accuracy_total_epoch += acc / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), max_grad_norm)

            optimizer.step()

        metrics_actor_critic = {f'{str(component)}/train/bc_loss': loss_total_epoch, f'{str(component)}/train/accuracy': accuracy_total_epoch}
        return [{'epoch': epoch, **metrics_actor_critic}]

    def bc_eval_actor_critic(self, epoch: int):
        self.agent.eval()

        metrics_actor_critic_eval = {}

        cfg_actor_critic = self.cfg.bc.actor_critic
        batch_num_samples = cfg_actor_critic.batch_num_samples
        sequence_length = cfg_actor_critic.burn_in + 1

        loss_total_eval = 0.0
        accuracy_total_eval = 0.0

        for _ in tqdm(range(cfg_actor_critic.steps_per_epoch), desc=f"Evaluating {str(self.agent.actor_critic)}", file=sys.stdout):
            batch = self.test_bc_dataset.sample_batch(batch_num_samples)
            assert batch['observations'].shape[1] == sequence_length
            batch = self._to_device(batch)

            loss, acc = self.agent.actor_critic.compute_bc_loss(batch)
            loss_total_eval += loss.item()
            accuracy_total_eval += acc
        
        loss_total_eval /= cfg_actor_critic.steps_per_epoch
        accuracy_total_eval /= cfg_actor_critic.steps_per_epoch
        
        metrics_actor_critic_eval = {f'{str(self.agent.actor_critic)}/eval/bc_loss': loss_total_eval, f'{str(self.agent.actor_critic)}/eval/accuracy': accuracy_total_eval}

        return [{'epoch': epoch, **metrics_actor_critic_eval}]
    
    def bc_eval_trajectory(self, epoch: int):
        self.agent.eval()

        metrics_actor_critic_eval_trajectory = {}

        cfg_actor_critic = self.cfg.bc.actor_critic
        num_eval_trajectories = cfg_actor_critic.num_eval_trajectories

        episode_lengths = []
        episode_rewards = []
        
        for _ in range(num_eval_trajectories):
            episode_env_reward, episode_length, episode_reward = self.agent.actor_critic.trajectory(self.bc_env)
            episode_lengths.append(episode_length)
            episode_rewards.append(episode_env_reward)
            print(f"Episode reward: {episode_reward}")
        
        avg_episode_length = sum(episode_lengths) / num_eval_trajectories
        avg_episode_reward = sum(episode_rewards) / num_eval_trajectories
        
        metrics_actor_critic_eval_trajectory = {f'{str(self.agent.actor_critic)}/eval/episode_length': avg_episode_length, f'{str(self.agent.actor_critic)}/eval/episode_reward': avg_episode_reward}

        return [{'epoch': epoch, **metrics_actor_critic_eval_trajectory}]
        

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer, sequence_length=1, sample_from_start=True, **cfg_tokenizer)
        self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component(self.agent.world_model, self.optimizer_world_model, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, tokenizer=self.agent.tokenizer, **cfg_world_model)
        self.agent.world_model.eval()

        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
        self.agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int, batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float], sequence_length: int, sample_from_start: bool, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sample_from_start)
                batch = self._to_device(batch)

                losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
    
    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        logging.info(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()

def main(cfg):
    trainer = Trainer(cfg)
    trainer.run()