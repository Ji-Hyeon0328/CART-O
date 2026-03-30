# import os
# import sys
# import torch
# import argparse
# from isaaclab.app import AppLauncher
# from typing import Dict

# # 1. 앱 런처 설정
# parser = argparse.ArgumentParser(description="Train Spot with CART Framework")
# parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
# AppLauncher.add_app_launcher_args(parser)
# args = parser.parse_args()

# launcher = AppLauncher(args)
# simulation_app = launcher.app

# # 2. 필요한 라이브러리 임포트
# from isaaclab.envs import ManagerBasedRLEnv
# #from isaaclab_carto.isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
# from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
# from spawn_spot import SpotActor # 같은 폴더에 있으므로 바로 임포트 가능

# from torch.utils.tensorboard import SummaryWriter
# import datetime
# from spawn_spot import SpotActor

# try:
#     import isaaclab_carto.isaaclab_carto.mdp as mdp
# except ImportError:
#     import isaaclab_carto.mdp as mdp
# def sanitize(x, nan=0.0, posinf=0.0, neginf=0.0):
#     return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

# def normalize_reward(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     return (x - x.mean()) / (x.std() + eps)

# def main():
#     # 환경 설정
#     env_cfg = CartoEnvCfg()
#     env_cfg.scene.num_envs = args.num_envs
#     env = ManagerBasedRLEnv(cfg=env_cfg)
    
#     # 텐서보드 기록기 초기화 (logs/carto_spot 폴더에 시간별로 저장)
#     log_dir = os.path.join("logs", "carto_spot", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     writer = SummaryWriter(log_dir)

#     # Figure 175 아키텍처 모델 생성
#     # num_proprio=36 (CART 논문 기준)
#     policy = SpotActor(num_proprio=36, num_map=187, num_actions=12).to(env.device)
#     optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

#     obs_dict, _ = env.reset()

#     print("-" * 50)
#     print("[INFO]: Figure 175 Hierarchical Training Started!")
#     print(f"[INFO]: Number of Envs: {env.num_envs}")
#     print("-" * 50)

#     # old_actions=None

#     while simulation_app.is_running():
        
#         obs_dict: Dict[str, torch.Tensor] = env.observation_manager.compute()
#         proprio = obs_dict["policy"].clone().detach() # [4, 36]
#         proprio=sanitize(proprio).float()
#         selector_aux = obs_dict["selector_aux"].clone().detach()

#         rgb = obs_dict["rgb_image"].clone().detach()
#         rgb=sanitize(rgb).float()
#         depth = obs_dict["depth_image"].clone().detach()
#         depth=sanitize(depth, nan=5.0, posinf=5.0, neginf=0.0).clamp(0.0, 5.0).float()

#         rgbd = torch.cat([rgb, depth], dim=-1)
        
#         ele_map = obs_dict["elevation_map"].clone().detach()
#         ele_map=sanitize(ele_map, nan=0.0, posinf=0.0, neginf=0.0).float()

#         cmd = env.command_manager.get_command("base_velocity").clone().detach()
#         cmd=sanitize(cmd).float()

#         selector_aux = selector_aux / selector_aux.abs().mean()
#         latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
#         R = policy.objective_selector(latent, cmd, proprio, selector_aux)

#         # Forward 실행
#         raw_actions = policy(proprio, rgbd, ele_map, cmd,R)
#         raw_actions = torch.clamp(raw_actions, min=-1.0, max=1.0)

#         dist = torch.distributions.Normal(raw_actions, 0.1)
#         actions=dist.rsample()
#         actions = torch.clamp(actions, min=-1.1, max=1.1)


#         # 환경 업데이트 및 보상 수집
#         obs_dict, rewards, terminated, truncated, extras = env.step(actions.detach())#env.step(actions)
        


#         # [핵심] 개별 보상 계산 및 가중치 결합 (미분 경로 생성)
#         r_v = mdp.reward_velocity(env)
#         r_s = mdp.reward_stability(env)
#         r_e = mdp.reward_energy(env)

#         r_v_n = normalize_reward(r_v)
#         r_s_n = normalize_reward(r_s)
#         r_e_n = normalize_reward(r_e)

#         #latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
#         #R = policy.objective_selector(torch.cat([latent, cmd], dim=-1)) # [num_envs, 3]

#         #weighted_reward = R[:, 0] * r_s_n + R[:, 1] * r_v_n + R[:, 2] * r_e_n
#         beta = torch.tensor([0.7, 0.2, 0.1], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
#         reward_vec = mdp.reward_tensor(env)          # [N, 3]
#         weighted_reward = mdp.combine_reward_components(beta, reward_vec)

#         reward_scale=1.0 #1e-3
#         log_prob = dist.log_prob(actions).sum(dim=-1)
#         entropy = -torch.sum(R * torch.log(R + 1e-8), dim=-1).mean()
#         loss = -(log_prob * weighted_reward.detach()).mean() * reward_scale - 0.01 * entropy


#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
#         optimizer.step()

#         # old_actions = actions.detach()


#         # [데이터 기록] 100 스텝마다 텐서보드에 저장
#         if env.common_step_counter % 100 == 0:
#             step = env.common_step_counter
            
#             # 1. 기본 훈련 지표 기록
#             writer.add_scalar("Train/Loss", loss.item(), step)
#             writer.add_scalar("Train/Avg_Reward", rewards.mean().item(), step)
            
#             # 2. Figure 175: Objective Selector 가중치(R) 기록
#             with torch.no_grad():
#                 #latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
#                 #R_weights = policy.objective_selector(torch.cat([latent, cmd], dim=-1))[0]
#                 latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
#                 R_weights = policy.objective_selector(latent, cmd, proprio, selector_aux)[0]
                
#                 writer.add_scalar("R_Weights/Stability", R_weights[0].item(), step)
#                 writer.add_scalar("R_Weights/Velocity", R_weights[1].item(), step)
#                 writer.add_scalar("R_Weights/Energy", R_weights[2].item(), step)
#                 writer.add_scalar("Component_Reward/Stability", r_s.mean().item(), env.common_step_counter)
#                 writer.add_scalar("Component_Reward/Velocity", r_v.mean().item(), env.common_step_counter)
#                 writer.add_scalar("Component_Reward/Energy", r_e.mean().item(), env.common_step_counter)

#             print(f"Step: {step} | Data logged to TensorBoard")
        
#         if env.common_step_counter % 1000 == 0:
#             torch.save(policy.state_dict(), f"logs/model_{env.common_step_counter}.pt")

#     env.close()

# if __name__ == "__main__":
#     main()

import os
import sys
import torch
import argparse
import datetime
from typing import Dict

from isaaclab.app import AppLauncher
from isaaclab_carto.utils.pseudo_expert_buffer import PseudoExpertBuffer, EpisodeRecord
# -----------------------------------------------------------------------------
# 1. CLI args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Spot with CART Framework")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument(
    "--use-objective-selector",
    action="store_true",
    help="Use learned objective selector. If false, use fixed beta preset."
)
parser.add_argument(
    "--beta-preset",
    type=str,
    default="balanced",
    choices=[
        "stability-heavy",
        "velocity-heavy",
        "energy-aware",
        "balanced",
        "cautious-balanced",
    ],
    help="Fixed reward-weight preset used when objective selector is disabled.",
)

parser.add_argument(
    "--use-pretrained-selector",
    action="store_true",
    help="Use a pretrained objective selector checkpoint."
)
parser.add_argument(
    "--selector-checkpoint",
    type=str,
    default="",
    help="Path to pretrained objective selector checkpoint."
)
parser.add_argument(
    "--freeze-selector",
    action="store_true",
    help="Freeze selector parameters during RL training."
)

parser.add_argument(
    "--min-success-length",
    type=int,
    default=50,
    help="Minimum episode length to consider an episode successful for pseudo-expert collection.",
)

parser.add_argument(
    "--terrain-id",
    type=str,
    default="unknown",
    help="Terrain identifier string for pseudo-expert logging.",
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Maximum env steps before stopping (0 = no limit)",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

# -----------------------------------------------------------------------------
# 2. Imports after app launch
# -----------------------------------------------------------------------------
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from spawn_spot import SpotActor
from torch.utils.tensorboard import SummaryWriter

try:
    import isaaclab_carto.isaaclab_carto.mdp as mdp
except ImportError:
    import isaaclab_carto.mdp as mdp


BETA_PRESETS = {
    # order: [velocity, slip, energy]
    "stability-heavy": [0.2, 0.7, 0.1],
    "velocity-heavy": [0.7, 0.2, 0.1],
    "energy-aware": [0.2, 0.2, 0.6],
    "balanced": [0.33, 0.33, 0.34],
    "cautious-balanced": [0.3, 0.5, 0.2],
}

def maybe_load_selector_checkpoint(policy, ckpt_path: str, freeze: bool = False):
    if not ckpt_path:
        print("[INFO] No selector checkpoint provided. Skipping load.")
        return False

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Selector checkpoint not found: {ckpt_path}")
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # case 1: full selector state_dict only
        try:
            policy.objective_selector.load_state_dict(ckpt, strict=False)
        except Exception:
            # case 2: wrapped training checkpoint
            if "selector" in ckpt:
                policy.objective_selector.load_state_dict(ckpt["selector"], strict=False)
            elif "model_state_dict" in ckpt:
                policy.objective_selector.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                raise

        print(f"[INFO] Loaded selector checkpoint from: {ckpt_path}")

        if freeze:
            for p in policy.objective_selector.parameters():
                p.requires_grad = False
            print("[INFO] Objective selector is frozen.")

        return True

    except Exception as e:
        print(f"[WARN] Failed to load selector checkpoint: {e}")
        return False

def sanitize(x, nan=0.0, posinf=0.0, neginf=0.0):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def normalize_reward(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)


def build_fixed_beta(num_envs: int, device: torch.device, preset_name: str) -> torch.Tensor:
    if preset_name not in BETA_PRESETS:
        raise ValueError(f"Unknown beta preset: {preset_name}")
    beta = torch.tensor(BETA_PRESETS[preset_name], dtype=torch.float32, device=device)
    beta = beta.unsqueeze(0).repeat(num_envs, 1)
    return beta

def estimate_success(
    terminated_flag: bool,
    truncated_flag: bool,
    episode_length: int,
    min_success_length: int = 50,
) -> bool:
    """
    Heuristic success rule for pseudo-expert collection.

    Current rule:
    - Success if the robot survived long enough, and
    - the episode ended by timeout/truncation OR did not catastrophically terminate early.

    This is a temporary heuristic until explicit goal-reaching / fall signals are added.
    """
    long_enough = episode_length >= min_success_length

    # Timeout/truncation is treated as success candidate
    if truncated_flag and long_enough:
        return True

    # If it did not truncate but also did not terminate catastrophically early,
    # allow long episodes as weak success candidates.
    if (not terminated_flag) and long_enough:
        return True

    return False

def main():
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    log_dir = os.path.join("logs", "carto_spot", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    policy = SpotActor(num_proprio=36, num_map=187, num_actions=12).to(env.device)
    #optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4)

    selector_loaded = False
    if args.use_pretrained_selector:
        selector_loaded = maybe_load_selector_checkpoint(
            policy,
            ckpt_path=args.selector_checkpoint,
            freeze=args.freeze_selector,
        )

    pseudo_buffer = PseudoExpertBuffer(capacity=5000)

    obs_dict, _ = env.reset()

    print("-" * 60)
    print("[INFO] CART-O training started")
    print(f"[INFO] num_envs = {env.num_envs}")
    print(f"[INFO] use_objective_selector = {args.use_objective_selector}")
    print(f"[INFO] use_pretrained_selector = {args.use_pretrained_selector}")
    print(f"[INFO] freeze_selector = {args.freeze_selector}")

    if args.use_pretrained_selector:
        print(f"[INFO] selector_checkpoint = {args.selector_checkpoint}")
        print(f"[INFO] selector_loaded = {selector_loaded}")

    if not args.use_objective_selector and not (args.use_pretrained_selector and selector_loaded):
        print(f"[INFO] fixed beta preset = {args.beta_preset} -> {BETA_PRESETS[args.beta_preset]}")
    print("-" * 60)

    episode_return_total = torch.zeros(env.num_envs, device=env.device)
    episode_return_v = torch.zeros(env.num_envs, device=env.device)
    episode_return_s = torch.zeros(env.num_envs, device=env.device)
    episode_return_e = torch.zeros(env.num_envs, device=env.device)
    episode_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    while simulation_app.is_running():
        obs_dict: Dict[str, torch.Tensor] = env.observation_manager.compute()

        proprio = sanitize(obs_dict["policy"].clone().detach()).float()
        selector_aux = sanitize(obs_dict["selector_aux"].clone().detach()).float()

        rgb = sanitize(obs_dict["rgb_image"].clone().detach()).float()
        depth = sanitize(
            obs_dict["depth_image"].clone().detach(),
            nan=5.0,
            posinf=5.0,
            neginf=0.0,
        ).clamp(0.0, 5.0).float()
        rgbd = torch.cat([rgb, depth], dim=-1)

        ele_map = sanitize(obs_dict["elevation_map"].clone().detach()).float()
        cmd = sanitize(env.command_manager.get_command("base_velocity").clone().detach()).float()

        selector_aux = selector_aux / (selector_aux.abs().mean() + 1e-6)

        # ---------------------------------------------------------------------
        # Beta selection
        # ---------------------------------------------------------------------
        latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)

        # if args.use_objective_selector:
        #     beta = policy.objective_selector(latent, cmd, proprio, selector_aux)
        # else:
        #     beta = build_fixed_beta(env.num_envs, env.device, args.beta_preset)

        use_selector_now = args.use_objective_selector or (args.use_pretrained_selector and selector_loaded)

        if use_selector_now:
            beta = policy.objective_selector(latent, cmd, proprio, selector_aux)
        else:
            beta = build_fixed_beta(env.num_envs, env.device, args.beta_preset)

        # ---------------------------------------------------------------------
        # Policy forward
        # ---------------------------------------------------------------------
        raw_actions = policy(proprio, rgbd, ele_map, cmd, beta)
        raw_actions = torch.clamp(raw_actions, min=-1.0, max=1.0)

        dist = torch.distributions.Normal(raw_actions, 0.1)
        actions = dist.rsample()
        actions = torch.clamp(actions, min=-1.1, max=1.1)

        obs_dict, rewards, terminated, truncated, extras = env.step(actions.detach())

        # ---------------------------------------------------------------------
        # Reward components: order = [velocity, slip, energy]
        # ---------------------------------------------------------------------
        reward_vec = mdp.reward_tensor(env)  # [N, 3]

        r_v = reward_vec[:, 0]
        r_s = reward_vec[:, 1]
        r_e = reward_vec[:, 2]

        reward_vec_n = torch.stack(
            [
                normalize_reward(r_v),
                normalize_reward(r_s),
                normalize_reward(r_e),
            ],
            dim=-1,
        )

        weighted_reward = mdp.combine_reward_components(beta, reward_vec_n)

        episode_return_total += weighted_reward.detach()
        episode_return_v += r_v.detach()
        episode_return_s += r_s.detach()
        episode_return_e += r_e.detach()
        episode_length += 1

        log_prob = dist.log_prob(actions).sum(dim=-1)

        # if args.use_objective_selector:
        #     beta_entropy = -torch.sum(beta * torch.log(beta + 1e-8), dim=-1).mean()
        # else:
        #     beta_entropy = torch.tensor(0.0, device=env.device)
        
        if use_selector_now:
            beta_entropy = -torch.sum(beta * torch.log(beta + 1e-8), dim=-1).mean()
        else:
            beta_entropy = torch.tensor(0.0, device=env.device)

        loss = -(log_prob * weighted_reward.detach()).mean() - 0.01 * beta_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        done = torch.logical_or(terminated, truncated)
        done_ids = torch.nonzero(done).squeeze(-1)

        for idx in done_ids.tolist():
            beta_i = beta[idx].detach().cpu().tolist()
            cmd_i = cmd[idx].detach().cpu().tolist()
            latent_i = latent[idx].detach().cpu().tolist()

            ep_len = int(episode_length[idx].item())
            ret_total = float(episode_return_total[idx].item())
            ret_v = float(episode_return_v[idx].item())
            ret_s = float(episode_return_s[idx].item())
            ret_e = float(episode_return_e[idx].item())

            mean_v = ret_v / max(ep_len, 1)
            mean_s = ret_s / max(ep_len, 1)
            mean_e = ret_e / max(ep_len, 1)

            # temporary heuristic
            #success = bool(not terminated[idx].item())
            ep_len = int(episode_length[idx].item())

            success = estimate_success(
                terminated_flag=bool(terminated[idx].item()),
                truncated_flag=bool(truncated[idx].item()),
                episode_length=ep_len,
                min_success_length=args.min_success_length,
            )

            record = EpisodeRecord(
                preset_name=args.beta_preset if not args.use_objective_selector else "selector",
                terrain_id=args.terrain_id,#"unknown",
                success=success,
                episode_length=ep_len,

                ended_by_termination=bool(terminated[idx].item()),
                ended_by_truncation=bool(truncated[idx].item()),
                
                beta=beta_i,
                command=cmd_i,
                return_total=ret_total,
                return_velocity=ret_v,
                return_slip=ret_s,
                return_energy=ret_e,
                mean_velocity=mean_v,
                mean_slip=mean_s,
                mean_energy=mean_e,
                policy_step=int(env.common_step_counter),
                timestamp=datetime.datetime.now().isoformat(),
                latent=latent_i,
            )
            pseudo_buffer.add(record)

            episode_return_total[idx] = 0.0
            episode_return_v[idx] = 0.0
            episode_return_s[idx] = 0.0
            episode_return_e[idx] = 0.0
            episode_length[idx] = 0

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------
        if env.common_step_counter % 100 == 0:
            step = env.common_step_counter

            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Avg_Env_Reward", rewards.mean().item(), step)
            writer.add_scalar("Train/Avg_Weighted_Reward", weighted_reward.mean().item(), step)

            writer.add_scalar("Component_Reward/Velocity", r_v.mean().item(), step)
            writer.add_scalar("Component_Reward/Slip", r_s.mean().item(), step)
            writer.add_scalar("Component_Reward/Energy", r_e.mean().item(), step)

            writer.add_scalar("Beta/Velocity", beta[:, 0].mean().item(), step)
            writer.add_scalar("Beta/Slip", beta[:, 1].mean().item(), step)
            writer.add_scalar("Beta/Energy", beta[:, 2].mean().item(), step)

            print(
                f"[step {step}] "
                f"loss={loss.item():.4f} | "
                f"w_reward={weighted_reward.mean().item():.4f} | "
                f"beta={beta.mean(dim=0).tolist()}"
            )

        if env.common_step_counter % 1000 == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_{env.common_step_counter}.pt")
            torch.save(policy.state_dict(), ckpt_path)
        
        if env.common_step_counter % 2000 == 0:
            pseudo_buffer.annotate_scores()
            buffer_path = os.path.join(log_dir, "pseudo_expert_buffer.json")
            pseudo_buffer.save_json(buffer_path)
            print(f"[INFO] saved pseudo buffer -> {buffer_path} ({len(pseudo_buffer)} episodes)")


        if args.max_steps > 0 and env.common_step_counter >= args.max_steps:
            print(f"[INFO] Reached max_steps={args.max_steps}, stopping.")
            break
    

    pseudo_buffer.annotate_scores()
    buffer_path = os.path.join(log_dir, "pseudo_expert_buffer.json")
    pseudo_buffer.save_json(buffer_path)
    print(f"[INFO] final pseudo buffer saved -> {buffer_path} ({len(pseudo_buffer)} episodes)")

    env.close()

    # For batch data collection, Isaac Sim GUI/plugin teardown can hang.
    # If max_steps > 0, exit the process immediately after saving.
    if args.max_steps > 0:
        print("[INFO] Batch/data-collection run finished. Exiting process directly.")
        os._exit(0)

    simulation_app.close()

if __name__ == "__main__":
    main()