# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from cleanrl.diayn.models_cont import Discriminator
import pickle


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int =81
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Walker2d-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    w_path: str  = "runs/checkpoints/env_phi_task/<your_task2_cont_run>/latest.pth"
    """from task2_cont.py"""
    model_path: str = "runs/checkpoints/maml/<your_sf_maml_cont_run>/latest.pth"
    """from sf_maml_cont.py"""
    disc_path: str = "runs/checkpoints/qtargetmaml/<unused>/latest.pth"
    """not used, discriminator loading is commented out below"""
    w_random: bool = False
    pretrained: bool = True
    n_skills_total: int = 25



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# # ALGO LOGIC: initialize agent here:
# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 120)
#         self.fc2 = nn.Linear(120, 120)
#         self.fc3 = nn.Linear(120, 32)

#     def forward(self, x, a):
#         x = torch.cat([x, a], 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        self.input_dim = state_dim + action_dim
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 32),  # 16-dim embedding
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.embedding(x)  # returns phi(s, a)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_mu = nn.Linear(84, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
    
class TaskVector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))

    def forward(self, phi_next):
        w_norm = self.w / (torch.norm(self.w) + 1e-8)
        return torch.matmul(phi_next, w_norm)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/cont3/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)

    # Inject SFNetworkBig first layer weights into actor's first layer and freeze it
    try:
        if args.pretrained:
            sf_ckpt = torch.load(args.model_path, map_location="cpu")
            if isinstance(sf_ckpt, dict) and "sfmeta_network_state_dict" in sf_ckpt:
                sf_sd = sf_ckpt["sfmeta_network_state_dict"]
            else:
                sf_sd = sf_ckpt if isinstance(sf_ckpt, dict) else {}

            sf_w = sf_sd.get("l1.weight", None)
            sf_b = sf_sd.get("l1.bias", None)
            if sf_w is not None and sf_b is not None:
                with torch.no_grad():
                    sf_w = sf_w.to(actor.fc1.weight.device)
                    sf_b = sf_b.to(actor.fc1.bias.device)
                    # Safe copy with overlap in case dims differ
                    rows = min(actor.fc1.weight.size(0), sf_w.size(0))
                    cols = min(actor.fc1.weight.size(1), sf_w.size(1))
                    actor.fc1.weight[:rows, :cols].copy_(sf_w[:rows, :cols])
                    actor.fc1.bias[:rows].copy_(sf_b[:rows])
                # Freeze actor's first layer
                actor.fc1.weight.requires_grad = False
                actor.fc1.bias.requires_grad = False
                print(f"Injected SFNetworkBig first layer into actor (rows={rows}, cols={cols}) and froze it")
            else:
                print("Warning: SF first layer keys not found; skipping injection")
    except Exception as e:
        print(f"Warning: failed to inject SF first layer into actor: {e}")

    qf1 = qf1.to(device)

    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    # Exclude frozen params from actor optimizer
    actor_optimizer = optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=args.learning_rate)
    
    state_dim = envs.single_observation_space.shape[0]

    # discriminator = Discriminator(state_dim, args.n_skills_total)
    # disc_ckpt = torch.load(args.disc_path, map_location="cpu")
    # discriminator.load_state_dict(disc_ckpt['disc_state_dict'])
    # #discriminator.load_state_dict(torch.load(args.disc_path)['disc_state_dict'])
    # discriminator = discriminator.to(device)

    w = torch.randn(32).to(device)
    w = w / (w.norm() + 1e-8)
    task_vector = TaskVector(32).to(device)
    checkpoint1 = torch.load(args.w_path)
    if(not args.w_random):
        task_vector.load_state_dict(checkpoint1["task_vector"])
    w = (task_vector.w / (torch.norm(task_vector.w) + 1e-8)).detach()

    # w = discriminator.q.weight[1].detach().to(device)
    # #w = torch.randn(32).to(device)
    # w = w / (w.norm() + 1e-8)


    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        #rewards = rewards #* 100  # <-- Scale up rewards by 100

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                #print("1", qf1_next_target.shape, w.shape)
                qvals_next = torch.einsum("bd,d->b", qf1_next_target, w)
                #qvals_next = torch.where(qvals_next > 300, torch.tensor(0.01, device=qvals_next.device), qvals_next)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qvals_next).view(-1)

            qf1_a_values = qf1(data.observations, data.actions)
            #print("2",qf1_a_values.shape, w.shape)
            qvals = torch.einsum("bd,d->b", qf1_a_values, w)
            # qvals = torch.where(qvals > 500, torch.tensor(0.5, device=qvals.device), qvals) # <-- Clip to -550 minimum
            qf1_loss = F.mse_loss(qvals, next_q_value)
            #qf1_loss = torch.clamp(qf1_loss, max=200)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                psi = qf1(data.observations, actor(data.observations))
                qvals1 = torch.einsum("bd,d->b", psi, w)
                #qvals1 = torch.where(qvals1 > 300, torch.tensor(0.01, device=qvals1.device), qvals1)
                actor_loss = -qvals1.mean()
                 # <-- Clip to -550 minimum
                # actor_loss = torch.clamp(actor_loss, min=-550)  # <-- Clip to -550 minimum
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qvals.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                #print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % 1000 == 0:
                writer.add_scalar("intrinsic_rewards", data.rewards.mean(), global_step)




    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        # print(f"mean reward: {total_rew/args.total_timesteps}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()