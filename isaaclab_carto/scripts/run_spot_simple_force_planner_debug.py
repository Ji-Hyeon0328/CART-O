# isaaclab_carto/scripts/run_spot_simple_force_planner_debug.py
#
# B4-b: simple MPC-shaped force planner debug.
#
# This is the last intermediate step before a real Python forceMPC port.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='CARTO Spot simple force planner debug')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--print_every', type=int, default=25)
parser.add_argument('--z_mode', type=str, default='conservative', choices=['conservative', 'aggressive'])
parser.add_argument('--vx', type=float, default=0.10)
parser.add_argument('--vy', type=float, default=0.0)
parser.add_argument('--wz', type=float, default=0.0)
parser.add_argument('--beta_preset', type=str, default='balanced', choices=['height', 'velocity', 'energy', 'balanced'])
parser.add_argument('--h_ref', type=float, default=0.67)
parser.add_argument('--mass', type=float, default=32.5)
parser.add_argument('--gravity', type=float, default=9.81)
parser.add_argument('--base_kp_h', type=float, default=40.0)
parser.add_argument('--base_kd_h', type=float, default=8.0)
parser.add_argument('--base_kp_vxy', type=float, default=10.0)
parser.add_argument('--residual_ratio', type=float, default=0.02)
parser.add_argument('--max_fz_per_foot', type=float, default=8.0)
parser.add_argument('--max_fxy_per_foot', type=float, default=2.0)
parser.add_argument('--smoothing_alpha', type=float, default=0.80)
parser.add_argument('--tau_scale', type=float, default=1.0)
parser.add_argument('--max_tau', type=float, default=3.0)
parser.add_argument('--force_sign', type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument('--linear_rows', type=str, default='0_3', choices=['0_3', '3_6'])
parser.add_argument('--ref_k', type=int, default=0)
parser.add_argument('--min_stance_legs', type=int, default=2)
parser.add_argument('--control_dt', type=float, default=0.02)
parser.add_argument('--spawn_z', type=float, default=0.60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg  # noqa: E402
from isaaclab_carto.envs.carto_effort_env_cfg import CartoEffortEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, build_spot_ref_params, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.theta_decoder import theta_decoder  # noqa: E402
from isaaclab_carto.lowlevel.theta_ref_mapper import theta_ref_mapper  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.simple_force_planner import SimpleForcePlannerState, make_simple_force_planner_torque  # noqa: E402


def patch_flat_safe_env(env_cfg: Any) -> None:
    try:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    except Exception as exc:
        print(f'[WARN] Could not set max_init_terrain_level: {exc}')
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        tg.size = (8.0, 8.0)
        tg.sub_terrains = {'flat': MeshPlaneTerrainCfg(proportion=1.0)}
        print('[INFO] Patched terrain generator to flat-only.')
    except Exception as exc:
        print(f'[WARN] Could not patch terrain generator to flat-only: {exc}')
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f'[INFO] Patched robot spawn z to {args.spawn_z}.')
    except Exception as exc:
        print(f'[WARN] Could not patch robot spawn z: {exc}')


def make_fixed_command(num_envs: int, device, dtype) -> torch.Tensor:
    u_cmd = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    u_cmd[:, 0] = args.vx
    u_cmd[:, 1] = args.vy
    u_cmd[:, 2] = args.wz
    return u_cmd


def make_fake_highlevel(num_envs: int, device, dtype):
    z_value = 0.0 if args.z_mode == 'conservative' else 1.0
    z_t = torch.full((num_envs,), z_value, device=device, dtype=dtype)
    a_hl = torch.tensor([0.20, -0.20, 0.30, -0.10], device=device, dtype=dtype)
    return z_t, a_hl.unsqueeze(0).repeat(num_envs, 1)


def make_beta(num_envs: int, device, dtype) -> torch.Tensor:
    if args.beta_preset == 'height':
        beta = torch.tensor([0.70, 0.20, 0.10], device=device, dtype=dtype)
    elif args.beta_preset == 'velocity':
        beta = torch.tensor([0.25, 0.65, 0.10], device=device, dtype=dtype)
    elif args.beta_preset == 'energy':
        beta = torch.tensor([0.25, 0.20, 0.55], device=device, dtype=dtype)
    elif args.beta_preset == 'balanced':
        beta = torch.tensor([0.45, 0.35, 0.20], device=device, dtype=dtype)
    else:
        raise ValueError(args.beta_preset)
    return beta.unsqueeze(0).repeat(num_envs, 1)


def advance_theta_phase_with_clock(theta, step: int, dt: float):
    T = theta.gait['T']
    phase_i = theta.gait['phase_i']
    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait['phase_i'] = torch.remainder(phase_i + phase_advance.unsqueeze(1), 1.0)
    return theta


def print_debug(step, robot, x_hat_before, u_cmd, beta_t, theta, ref, tau_cmd, info):
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame='world')
    print('\n' + '-' * 120)
    print(f'[SIMPLE FORCE PLANNER DEBUG] step={step}')
    print('-' * 120)
    print('[x_hat before env.step env0]')
    print('pos xyz      :', x_hat_before[env_id, 0:3].detach().cpu().numpy())
    print('rpy          :', x_hat_before[env_id, 3:6].detach().cpu().numpy())
    print('lin vel      :', x_hat_before[env_id, 6:9].detach().cpu().numpy())
    print('ang vel      :', x_hat_before[env_id, 9:12].detach().cpu().numpy())
    print('\n[x_hat after env.step env0]')
    print('pos xyz      :', x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print('rpy          :', x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print('lin vel      :', x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print('ang vel      :', x_hat_after[env_id, 9:12].detach().cpu().numpy())
    print('\n[u_cmd env0]')
    print(u_cmd[env_id].detach().cpu().numpy())
    print('\n[beta env0]')
    print(beta_t[env_id].detach().cpu().numpy(), f'preset={args.beta_preset}')
    print('\n[Theta env0]')
    print('z_mode       :', args.z_mode)
    print('T            :', float(theta.gait['T'][env_id].detach().cpu()))
    print('phase_i      :', theta.gait['phase_i'][env_id].detach().cpu().numpy())
    print('duty_i       :', theta.gait['duty_i'][env_id].detach().cpu().numpy())
    print('h_body_ref   :', float(theta.base['h_body_ref'][env_id].detach().cpu()))
    k = args.ref_k
    print(f'\n[Ref env0, k={k}]')
    print('S[:,k]       :', ref['S'][env_id, :, k].detach().cpu().numpy())
    print('phase[:,k]   :', ref['phase'][env_id, :, k].detach().cpu().numpy())
    print('\n[tau_simple_force_planner env0, action order]')
    print(tau_cmd[env_id].detach().cpu().numpy())
    print('tau stats:', summarize_torque(tau_cmd))
    print('\n[simple-force-planner info]')
    keys_to_show = [
        'beta_env0', 'kp_h_env0', 'kd_h_env0', 'kp_vxy_env0',
        'energy_scale_env0', 'smoothing_alpha_env0',
        'stance_mask_env0', 'num_stance_env0',
        'h_ref', 'h_mean', 'vz_mean', 'h_err_mean',
        'v_des_world_xy_env0', 'v_world_xy_env0', 'v_err_xy_env0',
        'fz_total_mean', 'fz_residual_total_mean', 'fz_per_stance_mean',
        'fxy_per_stance_env0', 'f_feet_env0',
        'tau_mean_abs', 'tau_max_abs', 'force_sign', 'linear_rows',
    ]
    for key in keys_to_show:
        if key in info:
            print(f'{key}: {info[key]}')
    if hasattr(robot.data, 'applied_torque'):
        applied = robot.data.applied_torque
        print('\n[applied_torque env0 after env.step]')
        print(applied[env_id].detach().cpu().numpy())
        print('applied stats:', summarize_torque(applied))
    print('-' * 120 + '\n')


def main() -> None:
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()
    robot = env.scene['robot']
    print_robot_debug_info(robot)
    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    params = build_spot_ref_params(device=device, dtype=dtype, dt=args.control_dt, horizon=20)
    planner_state = SimpleForcePlannerState()
    print('\n' + '=' * 120)
    print('[INFO] Starting Spot simple force planner debug')
    print('=' * 120)
    print(f'num_envs        : {env.num_envs}')
    print(f'steps           : {args.steps}')
    print(f'print_every     : {args.print_every}')
    print(f'z_mode          : {args.z_mode}')
    print(f'u_cmd           : [{args.vx}, {args.vy}, {args.wz}]')
    print(f'beta_preset     : {args.beta_preset}')
    print(f'h_ref           : {args.h_ref}')
    print(f'residual_ratio  : {args.residual_ratio}')
    print(f'max_fz_per_foot : {args.max_fz_per_foot}')
    print(f'max_fxy_per_foot: {args.max_fxy_per_foot}')
    print(f'smoothing_alpha : {args.smoothing_alpha}')
    print(f'max_tau         : {args.max_tau}')
    print(f'force_sign      : {args.force_sign}')
    print(f'linear_rows     : {args.linear_rows}')
    print(f'ref_k           : {args.ref_k}')
    print('=' * 120 + '\n')
    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        x_hat_before = make_x_hat(robot, velocity_frame='world')
        u_cmd = make_fixed_command(env.num_envs, device=device, dtype=dtype)
        z_t, a_hl = make_fake_highlevel(env.num_envs, device=device, dtype=dtype)
        beta_t = make_beta(env.num_envs, device=device, dtype=dtype)
        theta = theta_decoder(z_t=z_t, a_HL=a_hl, x_hat=x_hat_before, u_cmd=u_cmd, robot_name='spot')
        theta = advance_theta_phase_with_clock(theta, step=step, dt=args.control_dt)
        ref = theta_ref_mapper(theta=theta, x_hat=x_hat_before, u_cmd=u_cmd, params=params)
        tau_cmd, info = make_simple_force_planner_torque(
            robot=robot, ref=ref, x_hat=x_hat_before, u_cmd=u_cmd, beta_t=beta_t,
            planner_state=planner_state, h_ref=args.h_ref, mass=args.mass, gravity=args.gravity,
            base_kp_h=args.base_kp_h, base_kd_h=args.base_kd_h, base_kp_vxy=args.base_kp_vxy,
            residual_ratio=args.residual_ratio, max_fz_per_foot=args.max_fz_per_foot,
            max_fxy_per_foot=args.max_fxy_per_foot, smoothing_alpha=args.smoothing_alpha,
            tau_scale=args.tau_scale, max_tau=args.max_tau, linear_rows=args.linear_rows,
            force_sign=args.force_sign, use_k=args.ref_k, min_stance_legs=args.min_stance_legs)
        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)
        if step % args.print_every == 0:
            print_debug(step, robot, x_hat_before, u_cmd, beta_t, theta, ref, tau_cmd, info)
        if torch.any(terminated) or torch.any(truncated):
            print(f'[WARN] terminated/truncated at step={step}')
            print('terminated:', terminated.detach().cpu().numpy())
            print('truncated :', truncated.detach().cpu().numpy())
    env.close()
    simulation_app.close()


if __name__ == '__main__':
    main()
