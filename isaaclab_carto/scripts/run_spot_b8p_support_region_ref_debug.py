import os, sys, argparse
from typing import Any
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='CARTO/TRACER B8-p support-region reference debug')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--settle_steps', type=int, default=160)
parser.add_argument('--probe_steps', type=int, default=300)
parser.add_argument('--print_every', type=int, default=20)
parser.add_argument('--T', type=float, default=1.80)
parser.add_argument('--duty', type=float, default=0.78)
parser.add_argument('--phase_mode', type=str, default='crawl', choices=['crawl','trot'])
parser.add_argument('--control_dt', type=float, default=0.02)
parser.add_argument('--alpha', type=float, default=0.45)
parser.add_argument('--margin', type=float, default=0.030)
parser.add_argument('--max_shift_per_step', type=float, default=0.004)
parser.add_argument('--height_ref', type=float, default=0.54)
parser.add_argument('--roll_gain_from_y_error', type=float, default=0.0)
parser.add_argument('--pitch_gain_from_x_error', type=float, default=0.0)
parser.add_argument('--spawn_z', type=float, default=0.60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info
from isaaclab_carto.lowlevel.support_region_ref import SupportRegionRefConfig, compute_support_region_ref

FOOT_NAMES = ['fl_foot','fr_foot','hl_foot','hr_foot']
LEG_NAMES = ['LF','RF','LH','RH']

def patch_flat_safe_env(env_cfg: Any) -> None:
    try: env_cfg.scene.terrain.max_init_terrain_level = 0
    except Exception: pass
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1; tg.num_cols = 1; tg.size = (8.0, 8.0)
        tg.sub_terrains = {'flat': MeshPlaneTerrainCfg(proportion=1.0)}
        print('[INFO] Patched terrain generator to flat-only.')
    except Exception as exc: print(f'[WARN] Could not patch terrain generator: {exc}')
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f'[INFO] Patched robot spawn z to {args.spawn_z}.')
    except Exception as exc: print(f'[WARN] Could not patch robot spawn z: {exc}')

def get_foot_indices(robot):
    name_to_idx = {name:i for i,name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]

def gait_schedule(step: int, device, dtype):
    if args.phase_mode == 'crawl':
        phase0 = torch.tensor([0.00,0.25,0.50,0.75], device=device, dtype=dtype)
    else:
        phase0 = torch.tensor([0.00,0.50,0.50,0.00], device=device, dtype=dtype)
    phase = torch.remainder(phase0 + (step * args.control_dt) / max(args.T, 1e-6), 1.0)
    stance = (phase < args.duty).to(dtype)
    swing = 1.0 - stance
    progress = torch.clamp((phase - args.duty) / max(1.0 - args.duty, 1e-6), 0.0, 1.0) * swing
    return phase, stance, swing, progress

def print_debug(step, foot_pos, x_hat, phase, stance, swing, progress, out):
    print('\n' + '='*132)
    print(f'[B8-p SUPPORT REGION REF DEBUG] step={step}')
    print('='*132)
    print('[gait]')
    print('T:', args.T, 'duty:', args.duty, 'swing_time:', args.T*(1.0-args.duty))
    print('phase:', phase.detach().cpu().tolist())
    print('stance S:', stance.detach().cpu().tolist())
    print('swing:', swing.detach().cpu().tolist())
    print('swing_progress:', progress.detach().cpu().tolist())
    print('\n[current state env0]')
    print('base pos xyz:', x_hat[0,0:3].detach().cpu().numpy())
    print('base rpy:', x_hat[0,3:6].detach().cpu().numpy())
    for i,name in enumerate(LEG_NAMES): print(f'  {name}:', foot_pos[0,i].detach().cpu().numpy())
    print('\n[support region output env0]')
    print('stance_count:', int(out.stance_count[0].detach().cpu()))
    print('polygon_xy:', out.polygon_xy[0].detach().cpu().numpy())
    print('support_center_xy:', out.support_center_xy[0].detach().cpu().numpy())
    print('current_xy:', out.current_xy[0].detach().cpu().numpy())
    print('target_xy:', out.target_xy[0].detach().cpu().numpy())
    print('margin_to_edge:', float(out.margin_to_edge[0].detach().cpu()))
    print('swing_allowed:', bool(out.swing_allowed[0].detach().cpu()))
    print('base_ref [x,y,z,roll,pitch,yaw]:', out.base_ref[0].detach().cpu().numpy())
    print('base_ref_delta_xy:', (out.base_ref[0,0:2]-out.current_xy[0]).detach().cpu().numpy())
    print('='*132 + '\n')

def main():
    env_cfg = CartoEnvCfg(); env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)
    env = ManagerBasedRLEnv(cfg=env_cfg); env.reset()
    robot = env.scene['robot']; print_robot_debug_info(robot)
    device = robot.data.joint_pos.device; dtype = robot.data.joint_pos.dtype
    foot_indices = get_foot_indices(robot)
    cfg = SupportRegionRefConfig(alpha=args.alpha, margin=args.margin,
        max_shift_per_step=args.max_shift_per_step, height_ref=args.height_ref,
        roll_gain_from_y_error=args.roll_gain_from_y_error,
        pitch_gain_from_x_error=args.pitch_gain_from_x_error)
    print('\n' + '='*132); print('[INFO] Starting B8-p support-region reference debug'); print('cfg:', cfg); print('='*132)
    zero = torch.zeros((args.num_envs,12), device=device, dtype=dtype)
    for step in range(args.settle_steps):
        if not simulation_app.is_running(): break
        env.step(zero)
        if step % max(args.print_every,1) == 0:
            x = make_x_hat(robot, velocity_frame='world')
            print(f'[SETTLE] step={step} pos={x[0,0:3].detach().cpu().numpy()} rpy={x[0,3:6].detach().cpu().numpy()}')
    prev_base_ref = None
    for step in range(args.probe_steps):
        if not simulation_app.is_running(): break
        env.step(zero)
        x_hat = make_x_hat(robot, velocity_frame='world')
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        phase, stance, swing, progress = gait_schedule(step, device, dtype)
        stance_mask = stance.unsqueeze(0).repeat(args.num_envs, 1)
        out = compute_support_region_ref(foot_pos, x_hat[:,0:3], x_hat[:,3:6], stance_mask, prev_base_ref, cfg)
        prev_base_ref = out.base_ref.detach().clone()
        if step % max(args.print_every,1) == 0: print_debug(step, foot_pos, x_hat, phase, stance, swing, progress, out)
    env.close(); simulation_app.close()

if __name__ == '__main__': main()
