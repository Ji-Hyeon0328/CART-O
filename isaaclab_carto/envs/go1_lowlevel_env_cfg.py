"""Go1 torque-action environment config, v2.

Replace:
    source/isaaclab_carto/isaaclab_carto/envs/go1_lowlevel_env_cfg.py

Main fix in this version:
- Avoid assigning `robot = UNITREE_GO1_CFG.replace(...)` directly inside the
  config class when the type checker still thinks UNITREE_GO1_CFG may be None.
- Resolve Go1 config through a helper function first.
"""

from __future__ import annotations

from typing import Any, cast

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointEffortActionCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

# Use Isaac Lab built-in MDP functions.
import isaaclab.envs.mdp as mdp


def _load_go1_cfg() -> Any:
    """Load Go1 robot config from common Isaac Lab asset locations.

    Returning Any intentionally prevents the IDE/type checker from treating
    the result as Optional[None].
    """
    try:
        from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG as go1_cfg
        return go1_cfg
    except Exception as err_unitree_module:
        try:
            from isaaclab_assets.robots import UNITREE_GO1_CFG as go1_cfg
            return go1_cfg
        except Exception as err_robot_module:
            raise ImportError(
                "Could not import UNITREE_GO1_CFG. Your Isaac Lab installation may not expose "
                "the Unitree Go1 asset config. Try enabling/installing isaaclab_assets, or replace "
                "the robot config in this file with your own Go1 USD asset config. "
                f"Import errors: {err_unitree_module} | {err_robot_module}"
            ) from err_robot_module


GO1_CFG = _load_go1_cfg()


@configclass
class Go1LowLevelSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 4.0

    # Use cast(Any, ...) only to make static analysis happy.
    # Runtime object should be an ArticulationCfg and should have .replace().
    robot = cast(Any, GO1_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    #robot.init_state.pos = (0.0, 0.0, 0.34)
    # Start slightly higher to avoid initial ground penetration.
    # robot.init_state.pos = (0.0, 0.0, 0.34)
    # robot.init_state.joint_pos = {
    #     ".*_hip_joint": 0.0,
    #     ".*_thigh_joint": 0.75,
    #     ".*_calf_joint": -1.40,
    # }
    # robot.init_state.joint_vel = {
    #     ".*": 0.0,
    # }

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=1.0,
            num_rows=1,
            num_cols=1,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=1.0),
            },
        ),
        max_init_terrain_level=0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class Go1LowLevelActionsCfg:
    joint_effort = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class Go1LowLevelObservationsCfg:
    @configclass
    class PolicyObsCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyObsCfg = PolicyObsCfg()


@configclass
class Go1LowLevelRewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)


@configclass
class Go1LowLevelTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out)


@configclass
class Go1LowLevelEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 4
    episode_length_s: float = 20.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005)

    scene: Go1LowLevelSceneCfg = Go1LowLevelSceneCfg()
    observations: Go1LowLevelObservationsCfg = Go1LowLevelObservationsCfg()
    actions: Go1LowLevelActionsCfg = Go1LowLevelActionsCfg()
    rewards: Go1LowLevelRewardsCfg = Go1LowLevelRewardsCfg()
    terminations: Go1LowLevelTerminationsCfg = Go1LowLevelTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()  # type: ignore
