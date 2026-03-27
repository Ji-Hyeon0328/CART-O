import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import (
    ObservationGroupCfg, 
    ObservationTermCfg, 
    RewardTermCfg, 
    SceneEntityCfg,
    TerminationTermCfg
)

try:
    import isaaclab_carto.isaaclab_carto.mdp as mdp
except ImportError:
    import isaaclab_carto.mdp as mdp

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sensors.camera.camera_cfg import PinholeCameraCfg

from isaaclab.sensors import TiledCameraCfg

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
#from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg, HfPyramidSlopedTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

from isaaclab.managers import EventTermCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfSteppingStonesTerrainCfg,
)
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg


@configclass
class CartoSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 10.0
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/BostonDynamics/spot/spot.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), # 착지 충격을 줄이기 위해 높이 조절
            joint_pos={
                ".*hx": 0.0, ".*hy": 0.8, ".*kn": -1.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=180.0, # 근력 강화
                damping=5.0,    # 진동 억제
            ),
        },
    )
    light = AssetBaseCfg(
    prim_path="/World/light",
    spawn=sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    ),
    )

    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground", # 지형이 생성될 경로
    #     terrain_type="generator",   # 절차적 생성기를 사용함을 명시
    #     terrain_generator=TerrainGeneratorCfg(
    #         size=(8.0, 8.0),
    #         border_width=20.0,
    #         num_rows=16,
    #         num_cols=16,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.005,
    #         slope_threshold=0.75,
    #         use_cache=False,
    #         sub_terrains={
    #             "rough": HfRandomUniformTerrainCfg(
    #                 proportion=0.3,
    #                 noise_range=(0.02, 0.1), 
    #                 noise_step=0.01,
                    
    #             ),
    #             "hills": HfPyramidSlopedTerrainCfg(
    #                 proportion=0.3,
    #                 slope_range=(0.0, 0.4),
    #                 #noise_step=0.01,
    #                 platform_width=2.0
    #             ),
    #             "flat_ice_candidate": MeshPlaneTerrainCfg(
    #                 proportion=0.4,
    #             ),
    #         },
    #     ),
    #     max_init_terrain_level=5, # 초기 지형 난이도 설정
    # )

    terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    use_terrain_origins=True, # False
    env_spacing=10.0,
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=1.0,
        num_rows=16,
        num_cols=16,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        color_scheme="height",
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(
                proportion=0.15,
            ),
            "mild_rough": HfRandomUniformTerrainCfg(
                proportion=0.20,
                noise_range=(0.01, 0.03),
                noise_step=0.01,
            ),
            "hard_rough": HfRandomUniformTerrainCfg(
                proportion=0.20,
                noise_range=(0.05, 0.15),
                noise_step=0.01,
            ),
            "slope": HfPyramidSlopedTerrainCfg(
                proportion=0.15,
                slope_range=(0.10, 0.35),
                platform_width=2.0,
            ),
            "stairs": HfPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.18),
                step_width=0.30,
                platform_width=2.0,
            ),
            "obstacles": HfDiscreteObstaclesTerrainCfg(
                proportion=0.15,
                obstacle_width_range=(0.2, 0.6),
                obstacle_height_range=(0.05, 0.20),
                num_obstacles=18,
                platform_width=2.0,
            ),
        },
    ),
    max_init_terrain_level=15,
    )

    # [추가] RGB-D 데이터를 위한 카메라 센서
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        # [수정] UsdFileCfg 대신 PinholeCameraCfg를 사용합니다.
        spawn=PinholeCameraCfg(), 
    )

    # # [추가] 지형 스캔을 위한 레이캐스터 센서
    # height_scanner: RayCasterCfg = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), # 위에서 아래로 쏨
    #     attach_yaw_only=True,
    #     pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # 로봇 주변 1.6m x 1.0m 스캔 # type: ignore
    #     debug_vis=True, # 시뮬레이션 창에서 스캔 포인트 확인 가능
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner: RayCasterCfg = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # type: ignore
    debug_vis=False, #True
    mesh_prim_paths=["/World/ground"],
)

@configclass
class CartoActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=1.0
    )
    
@configclass
class CartoEventCfg:
    """학습 중 지형 마찰력을 랜덤하게 변경 (빙판 vs 풀밭 시뮬레이션)"""
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset", # 환경이 리셋될 때마다 변경
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.1, 1.0), # 0.1(빙판) ~ 1.0(잔디)
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 0.0),
        },
    )
events: CartoEventCfg = CartoEventCfg()

@configclass
class CartoObservationsCfg:
    @configclass
    class ProprioObsCfg(ObservationGroupCfg): 
        """Figure 175 & CART 논문 기준 36차원 p_t 구성"""
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        # [수정] params를 추가하여 mdp 함수와 연결합니다.
        joint_efforts = ObservationTermCfg(
            func=mdp.joint_efforts, 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_height = ObservationTermCfg(
            func=mdp.base_height, 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        feet_slip = ObservationTermCfg(
        func=mdp.feet_slip_summary,#feet_slip_per_foot,
        params={"asset_cfg": SceneEntityCfg("robot")}
        )   

    # 1. 기존의 48차원 수치 데이터 그룹
    # @configclass
    # class PolicyObsCfg(ObservationGroupCfg): 
    #     base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
    #     base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
    #     projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
    #     velocity_commands = ObservationTermCfg(
    #         func=mdp.generated_commands, 
    #         params={"command_name": "base_velocity"}
    #     )
    #     joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    #     joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
    #     last_action = ObservationTermCfg(func=mdp.last_action)

    ## policy: PolicyObsCfg = PolicyObsCfg()
    # policy: ProprioObsCfg = ProprioObsCfg()
    @configclass
    class PolicyObsCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)           # 12
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)           # 12
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)         # 3
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)         # 3
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity) # 3
        base_height = ObservationTermCfg(
            func=mdp.base_height,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )                                                                # 1
        feet_slip = ObservationTermCfg(
            func=mdp.feet_slip_summary,#feet_slip_per_foot,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
    
    policy: PolicyObsCfg = PolicyObsCfg()
    #policy: ProprioObsCfg = ProprioObsCfg()
    
    @configclass
    class SelectorObsCfg(ObservationGroupCfg):
        reference_friction = ObservationTermCfg(
            func=mdp.reference_friction,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )   # 예: 4차원

        base_height = ObservationTermCfg(
            func=mdp.base_height,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )   # 1차원

        feet_slip = ObservationTermCfg(
            func=mdp.feet_slip_summary,#feet_slip_per_foot,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
    selector_aux: SelectorObsCfg = SelectorObsCfg()
    

    # 2. [수정] 지형 맵 그룹 (Figure 175: Map 입력)
    @configclass
    class MapObsCfg(ObservationGroupCfg):
        map = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")}
        )
    elevation_map: MapObsCfg = MapObsCfg()

    # 3. [수정] RGB 이미지 그룹 (Figure 175: RGB-D 입력)
    @configclass
    class RGBObsCfg(ObservationGroupCfg):
        rgb = ObservationTermCfg(
            func=mdp.camera_rgb,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera")}
        )
    rgb_image: RGBObsCfg = RGBObsCfg()

    # 4. [수정] Depth 이미지 그룹 (Figure 175: RGB-D 입력)
    @configclass
    class DepthObsCfg(ObservationGroupCfg):
        depth = ObservationTermCfg(
            func=mdp.camera_depth,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera")}
        )
    depth_image: DepthObsCfg = DepthObsCfg()

@configclass
class CartoRewardsCfg:
    """Figure 175: 보상 설정"""
    total_reward = RewardTermCfg(
        func=mdp.carto_reward_total, 
        weight=1.0,
        params={
            "beta_weights": torch.tensor([[1.0, 0.0, 0.0]], device="cuda:0"), 
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

@configclass
class CartoCommandsCfg:
    """로봇 명령 설정"""
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

@configclass
class CartoTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, params={})
    
    #terminate when it sits down.
    base_height_termination = TerminationTermCfg(
        func=mdp.base_height_below_threshold, # 이름 주의!
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "threshold": 0.25 # 이제 threshold를 정상적으로 인식합니다.
        }
    )


@configclass
class CartoEnvCfg(ManagerBasedRLEnvCfg):
    # 환경 파라미터
    decimation: int = 4
    episode_length_s: float = 20.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005)

    # 매니저 연결 (변수 이름이 Isaac Lab 표준 이름과 일치해야 합니다)
    commands: CartoCommandsCfg = CartoCommandsCfg()
    scene: CartoSceneCfg = CartoSceneCfg()
    observations: CartoObservationsCfg = CartoObservationsCfg()
    rewards: CartoRewardsCfg = CartoRewardsCfg()
    actions: CartoActionsCfg = CartoActionsCfg()
    terminations: CartoTerminationsCfg = CartoTerminationsCfg()

    def __post_init__(self):
        # 부모 클래스 초기화가 누락되면 reset() 시 AttributeError가 발생합니다.
        super().__post_init__() # type: ignore