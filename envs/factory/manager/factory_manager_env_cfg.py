from dataclasses import MISSING
import torch
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
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.lab.sensors import TiledCameraCfg, ImuCfg


from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


import envs.factory.manager.mdp.rewards as fac_mdp_rew
import envs.factory.manager.mdp.observations as fac_mdp_obs
import envs.factory.manager.mdp.events as fac_mdp_events
import envs.factory.manager.mdp.curriculum as fac_mdp_cric

from envs.factory.manager.factory_manager_task_cfg import *
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

from sensors.dense_pose_sensor.dense_pose_sensor import DummySensorCfg, DensePoseSensorCfg
from sensors.force_torque_sensor.force_torque_cfg import ForceTorqueSensorCfg
##
# Scene definition
##

@configclass
class FactoryManagerSceneCfg(InteractiveSceneCfg):
    """Configuration for the peg and hole scene with a robot, a hole and a peg.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    #robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
            
            # this actuator config makes it so that setting joint_effort_target goes straight to the actuator
            # as the actuator stiffness and damping are zero.  Factory uses operational spaces gains for the control
            #"panda_arm1": ImplicitActuatorCfg(
            #    joint_names_expr=["panda_joint[1-4]"],
            #    stiffness=0.0,
            #    damping=0.0,
            #    friction=0.0,
            #    armature=0.0,
            #    effort_limit=87,
            #    velocity_limit=124.6,
            #),
            #"panda_arm2": ImplicitActuatorCfg(
            #    joint_names_expr=["panda_joint[5-7]"],
            #    stiffness=0.0,
            #    damping=0.0,
            #    friction=0.0,
            #    armature=0.0,
            #    effort_limit=12,
            #    velocity_limit=149.5,
            #),
            
        },
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0, 0], rot=[0.70711, 0, 0, 0.70711]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )                               

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.4]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    #task_cfg = PegInsert()

    fixed_asset: ArticulationCfg = PegInsert().fixed_asset

    held_asset: ArticulationCfg = PegInsert().held_asset

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.35), 
            rot=(0.6123724, 0.3535534, 0.3535534, 0.6123724,), 
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=0.05, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=240,
        height=180,
        debug_vis = False,
    )

    force_torque_sensor: ForceTorqueSensorCfg = ForceTorqueSensorCfg(
        prim_path="/World/envs/env_.*/Robot",
        history_length=1,
        debug_vis = False,
        update_period = 0,
    )
    


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_fingertip_centered",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale= 0.05,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    
    #arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
    #    asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True
    #)
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # fingertip to fixed
        # fingertip to held
        # fingertip vel
        # previous action
        # force
        """
        fingertip_pos = ObsTerm(
            func= fac_mdp_obs.fingertip_pos
        )

        fingertip_quat = ObsTerm(
            func = fac_mdp_obs.fingertip_quat
        )
        """

        fingertip_pos_rel_held = ObsTerm(
            func = fac_mdp_obs.robot_held_relative_pos
        )

        fingertip_qual_rel_held = ObsTerm(
            func = fac_mdp_obs.robot_held_relative_quat
        )

        ee_linvel = ObsTerm(
            func = fac_mdp_obs.ee_linvel
        )

        ee_angvel = ObsTerm(
            func = fac_mdp_obs.ee_angvel
        )

        fingertip_pos_rel_fixed = ObsTerm(
            func = fac_mdp_obs.robot_fixed_relative_pos
        )

        fingertip_qual_rel_fixed = ObsTerm(
            func = fac_mdp_obs.robot_fixed_relative_quat
        )
        
        force_torque_reading = ObsTerm(
            func=fac_mdp_obs.force_torque_sensor_scaled
        )

        prev_action = ObsTerm(func=mdp.last_action)

        #hole_pose = ObsTerm(
        #    func=fac_mdp_obs.fixed_asset_pose
        #)


        """ What factory Uses by default        
            #"fingertip_pos": self.fingertip_midpoint_pos,
            #"fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            #"fingertip_quat": self.fingertip_midpoint_quat,
            #"ee_linvel": self.ee_linvel_fd,
            #"ee_angvel": self.ee_angvel_fd,
            #"prev_actions": prev_actions,
        """

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class InfoCfg(ObsGroup):
        """Observations for information tracking"""

        dmg_force = ObsTerm(
            func=fac_mdp_obs.force_torque_sensor
        )

        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        joint_acc = ObsTerm(func=fac_mdp_obs.joint_acc_rel)

        img = ObsTerm(func=fac_mdp_obs.camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    info: InfoCfg = InfoCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    set_body_inertias = EventTerm(
        func=fac_mdp_events.set_body_inertias,
        mode="startup"
    )

    init_memory = EventTerm(
        func=fac_mdp_events.init_tensors,
        mode="startup",
        params={
            "task_cfg": PegInsert()
        }
    )

    #init_imu = EventTerm(
    #    func=fac_mdp_events.init_imu,
    #    mode="startup"
    #)

    set_default_dynamics_params = EventTerm(
        func=fac_mdp_events.set_default_dynamics_parameters,
        mode="startup"
    )

    #init_force_torque_sensor = EventTerm(
    #    func=fac_mdp_events.init_ft_sensor,
    #    mode="startup"
    #)

    reset_assets = EventTerm(
        func = fac_mdp_events.set_assets_to_default_pose,
        mode="reset"
    )

    reset_franka = EventTerm(
        func=fac_mdp_events.set_franka_to_default_pose,
        mode="reset"
    )


    randomize_fixed_asset_init = EventTerm(
        func=fac_mdp_events.reset_fixed_asset,
        mode="reset",
        #params={
        #    #"pose_range": {"x": (0.565, 0.665), "y": (-0.1, 0.1), "yaw": (3.14159/2 - 3.14159/3, 3.14159/2 + 3.14159/3)},
        #    "pose_range": {"x": (0.615, 0.615)},
        #    "velocity_range": {},
        #    #"asset_cfg": SceneEntityCfg("hole", body_names="Hole"),
        #},
    )

    randomize_franka_arm = EventTerm(
        func = fac_mdp_events.reset_franka_above_fixed,
        mode="reset"
    )

    randomize_held_asset = EventTerm(
        func=fac_mdp_events.reset_held_asset,
        mode="reset"
    )

    
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    keypoint_baseline = RewTerm(
        func=fac_mdp_rew.keypoint_reward,
        params={
            "a" : 5.0,
            "b" : 4.0
        },
        weight=1.0
    )

    keypoint_coarse = RewTerm(
        func=fac_mdp_rew.keypoint_reward,
        params={
            "a" : 50.0,
            "b" : 2.0
        },
        weight=1.0
    )

    keypoint_fine = RewTerm(
        func=fac_mdp_rew.keypoint_reward,
        params={
            "a" : 100.0,
            "b" : 0.0
        },
        weight=1.0
    )

    engaged = RewTerm(
        func=fac_mdp_rew.currently_inrange,
        params={
            "success_threshold" : 0.9,
            "check_rot" : False
        },
        weight=1.0
    )

    success = RewTerm(
        func=fac_mdp_rew.currently_inrange,
        params={
            "success_threshold" : 0.04,
            "check_rot" : False
        },
        weight=1.0
    )

    # factory includes these but seems to be zero reward
    l2_action_penalty = RewTerm(
        func=mdp.action_l2,
        weight=0.0
    )

    l2_action_grad_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.0
    )

from omni.isaac.lab.envs import ManagerBasedRLEnv
def random_stop(env: ManagerBasedRLEnv) -> torch.Tensor:
    rand = torch.rand((env.num_envs ), dtype=torch.float32, device=env.device)
    print(rand < 0.05)
    if torch.any(rand < 0.05):
        print("Resetting")
    return rand < 0.05

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #random = DoneTerm(func=random_stop, time_out=False)



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    init_height_sampling = CurrTerm(
        func=fac_mdp_cric.update_z_low
    )

##
# Environment configuration
##

@configclass
class FactoryManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    """scene: PegHoleTabletopSceneCfg = PegHoleTabletopSceneCfg(
        num_envs=num_envs, 
        env_spacing=2.5, 
        replicate_physics=False
    )"""
    num_envs = 2
    replicate_physics = True
    env_spacing = 2.5
    decimation = 20
    episode_length_s = 5.0

    # Basic settings
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt= 1 / 200.0,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
            gpu_collision_stack_size=150000000
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.sim.render_interval = self.decimation

        self.recording = False
        # move this scene here so that the correct number of envs
        # is present in the post_init function, allowing for intelligent 
        # peg and hole generations
        self.scene = FactoryManagerSceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing, 
            replicate_physics=self.replicate_physics
        )