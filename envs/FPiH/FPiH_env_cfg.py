from dataclasses import MISSING

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    DeformableObjectCfg, 
)
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.envs import mdp
import envs.FPiH.mdp.events as fpih_mdp_events
import envs.FPiH.mdp.observations as fpih_mdp_obs
import envs.FPiH.mdp.rewards as fpih_mdp_rew
from omni.isaac.lab.sensors import ContactSensorCfg

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.lab.sensors import TiledCameraCfg

#from . import mdp

##
# Scene definition
##


@configclass
class PegHoleTabletopSceneCfg(InteractiveSceneCfg):
    """Configuration for the peg and hole scene with a robot, a hole and a peg.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    #ee_frame: FrameTransformerCfg = MISSING
    #hole_frame: FrameTransformerCfg = MISSING
    #peg_end_frame: FrameTransformerCfg = MISSING

        
    # peg and hole: will be populated by agent env cfg
    #peg: RigidObjectCfg | DeformableObjectCfg = MISSING

    #peg_contact_force: ContactSensorCfg = MISSING


    #hole: RigidObject = MISSING
    #hole: RigidObjectCfg | RigidObjectCollectionCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.5, 0.0, 1.25), 
            rot=(0.6123724, 0.3535534, 0.3535534, 0.6123724,), 
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=240,
        height=180,
        debug_vis = True,
    )


    robot_av = None

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    """object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )"""

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    #gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=fpih_mdp_obs.object_position_in_robot_root_frame)
        force_torque = ObsTerm(func=fpih_mdp_obs.force_torque_sensor)

        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        #actions = ObsTerm(func=mdp.last_action)

        peg_pose = ObsTerm(
            func=fpih_mdp_obs.frame_in_robot_frame,
            params={
                "frame_name":"peg_end_frame"
            }
        )

        hole_pose = ObsTerm(
            func=fpih_mdp_obs.frame_in_robot_frame,
            params={
                "frame_name":"hole_frame"
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class InfoCfg(ObsGroup):
        """Observations for information tracking"""

        dmg_force = ObsTerm(
            func=fpih_mdp_obs.force_torque_sensor
        )

        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        joint_acc = ObsTerm(func=fpih_mdp_obs.joint_acc_rel)

        img = ObsTerm(func=fpih_mdp_obs.camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    info: InfoCfg = InfoCfg()

#TODO: Seperate out reset functions
    # remove constants
@configclass
class EventCfg:
    """Configuration for events."""
    """peg_physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("peg"),
          "static_friction_range": (2.0, 2.0),
          "dynamic_friction_range": (2.0, 2.0),
          "restitution_range": (1.0, 1.0),
          "num_buckets": 250,
      },
    )"""

    init_ft_sensor = EventTerm(func=fpih_mdp_events.init_ft_sensor, mode="startup")

    fix_peg_to_robot = EventTerm(
        func=fpih_mdp_events.afix_peg_to_robot,
        mode="startup"
    )

    init_offsets = EventTerm(
        func=fpih_mdp_events.init_default_hole_pos,
        mode="startup"
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    #reset_robot_joint_offset = EventTerm(
    #    func=mdp.reset_joints_by_offset,
    #    mode="reset",
    #    params={
    #        "position_range": (-0.02, 0.02),
    #        "velocity_range": (0.0, 0.0)
    #    },
    #)

    reset_hole_position = EventTerm(
        func=fpih_mdp_events.reset_hole,
        mode="reset",
        params={
            "pose_range": {"x": (0.565, 0.665), "y": (-0.1, 0.1), "yaw": (3.14159/2 - 3.14159/3, 3.14159/2 + 3.14159/3)},
            #"pose_range": {"x": (0.615, 0.615)},
            "velocity_range": {},
            #"asset_cfg": SceneEntityCfg("hole", body_names="Hole"),
        },
    )

    #reset_peg = EventTerm(
    #    func=fpih_mdp_events.reset_peg,
    #    mode="reset"
    #)

    reset_robot = EventTerm(
        func = fpih_mdp_events.reset_robot,
        mode="reset"
    )

#TODO: Port rewards from maniskill
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reward being rightside up
    alignment = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("peg")
        },
        weight= -1.0
    )

    # reward l2 dist
    dist_to_goal = RewTerm(
        func=fpih_mdp_rew.peg_hole_xy,
        weight=5.0
    )

    # punish failure
    failure = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10.0,
        params = {
            "term_keys":[
                "peg_broke",
                "object_dropping"
            ]
        }
    )
    # reward success
    success = RewTerm(
        func = mdp.is_terminated_term,
        weight=100,
        params={
            "term_keys":["success"]
        }
    )
    # action penalty
    #action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    #joint_vel = RewTerm(
    #    func=mdp.joint_vel_l2,
    #    weight=-1e-4,
    #    params={"asset_cfg": SceneEntityCfg("robot")},
    #)



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": -0.025, "asset_cfg": SceneEntityCfg("peg")}
    )

    peg_broke = DoneTerm(
        func=mdp.terminations.illegal_contact,
        params={
            "threshold": 1000000.0,
            "sensor_cfg": SceneEntityCfg("peg_contact_force")}
    )

    success = DoneTerm(
        func=fpih_mdp_obs.is_peg_inserted
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    """
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
    """


##
# Environment configuration
##

@configclass
class FragilePegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    """scene: PegHoleTabletopSceneCfg = PegHoleTabletopSceneCfg(
        num_envs=num_envs, 
        env_spacing=2.5, 
        replicate_physics=False
    )"""
    num_envs = 2
    replicate_physics = False
    env_spacing = 2.5
    # Basic settings
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        # move this scene here so that the correct number of envs
        # is present in the post_init function, allowing for intelligent 
        # peg and hole generations
        self.scene = PegHoleTabletopSceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing, 
            replicate_physics=self.replicate_physics
        )
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625