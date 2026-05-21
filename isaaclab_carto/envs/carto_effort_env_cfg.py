# isaaclab_carto/envs/carto_effort_env_cfg.py
#
# Effort-action environment config for CARTO / TRACER Spot B-step debugging.
#
# This keeps the original CartoEnvCfg scene/observations/rewards/terminations,
# but replaces JointPositionActionCfg with JointEffortActionCfg.

from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointEffortActionCfg

from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg


@configclass
class CartoEffortActionsCfg:
    """Torque / effort action interface."""

    joint_effort = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class CartoEffortEnvCfg(CartoEnvCfg):
    """CartoEnvCfg with torque/effort action instead of joint-position action."""

    actions: CartoEffortActionsCfg = CartoEffortActionsCfg()

    def __post_init__(self):
        super().__post_init__()  # type: ignore
