from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollectionCfg, AssetBaseCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.envs import mdp


from omni.isaac.lab.assets.articulation import ArticulationCfg
import omni.isaac.core.utils.prims as prim_utils
from envs.FPiH.FPiH_env_cfg import PegHoleTabletopSceneCfg, FragilePegInHoleEnvCfg
#from envs.FPiH import mdp

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


from omni.isaac.core.articulations import ArticulationView
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import ContactSensorCfg

import numpy as np
import torch

@configclass
class FrankaFragilePegInHoleCfg(FragilePegInHoleEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.robot.spawn.usd_path = "franka_with_block.usd"
        #self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #print("type:", type(self.scene.robot))
        
        self.scene.robot.init_state.joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 3.14159 / 8, #-0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -3.14159 * 5 / 8, #-2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.14159 * 3 / 4, #3.037,
                "panda_joint7": 3.14159 / 4, #0.741,
                "panda_finger_joint.*": 0.00,
        
        }
        
        self.scene.robot.actuators['panda_hand'].effort_limit = 1600.0
        
        self.x_offset = 0.0 #0.615 # THIS HAS BEEN Depreciated, can be handled by the reset spawning variables 
        
        for idx in range(self.scene.num_envs):
            prim_utils.create_prim(
                f"/World/envs/env_{idx}/Hole", 
                "Xform", 
                translation=[self.x_offset,0,0.0],
                orientation=[0.0,0.0,0.0,0.0]
            )
        
        # Set actions for the specific robot type (franka)
        #self.actions.arm_action = mdp.JointPositionActionCfg(
        #    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        #)
        #self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        #    asset_name="robot", joint_names=["panda_joint.*"], use_zero_offset=True, scale=1.0
        #)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True
        )

        """
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        """

        peg_cfgs, box_cfgs, self.peg_offsets, self.hole_init_poses = self.get_asset_cfgs( 
            radii = (0.015, 0.025),
            lengths = (0.085, 0.125),
            clearance = 0.003
        )
        
        self.scene.peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.5, 0.0125], rot=[0.707, 0.707, 0, 0]
            ),
            spawn = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=peg_cfgs,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, 
                    solver_velocity_iteration_count=16,
                    #angular_damping = 0.75,
                    #kinematic_enabled=True
                ),
                
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    #torsional_patch_radius=1.0,
                    #contact_offset = 0.002,
                    #rest_offset = 0.001
                    
                ),
                activate_contact_sensors = True,
                random_choice = False
            )
        )

        self.have_peg_pose = False
        self.recording = False

        self.scene.peg_end_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            #visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Peg",
                    name="peg_end_frame",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        self.scene.peg_end_frame.visualizer_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
        self.scene.peg_end_frame.visualizer_cfg.prim_path="/Visuals/peg_end_frame"
        
        
        self.scene.peg_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
        )
        
        self.panda_ee_offset = 0.1034
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            #visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, self.panda_ee_offset],
                    ),
                ),
            ],
        )
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.scene.ee_frame.visualizer_cfg.prim_path="/Visuals/ee_frame"
        
        
        self.scene.hole = RigidObjectCollectionCfg(
            rigid_objects = {
                "hole_top_left": RigidObjectCfg(
                    prim_path = "{ENV_REGEX_NS}/Hole/top_left",
                    spawn = sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg = box_cfgs,
                        random_choice = False
                    ),
                    init_state=self.hole_init_poses[0][0]
                ),
                "hole_top_right": RigidObjectCfg(
                    prim_path = "{ENV_REGEX_NS}/Hole/top_right",
                    spawn = sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg = box_cfgs,
                        random_choice = False
                    ),
                    init_state=self.hole_init_poses[0][1]
                ),
                "hole_bot_left": RigidObjectCfg(
                    prim_path = "{ENV_REGEX_NS}/Hole/bot_left",
                    spawn = sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg = box_cfgs,
                        random_choice = False
                    ),
                    init_state=self.hole_init_poses[0][2]
                ),
                "hole_bot_right": RigidObjectCfg(
                    prim_path = "{ENV_REGEX_NS}/Hole/bot_right",
                    spawn = sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg = box_cfgs,
                        random_choice = False
                    ),
                    init_state=self.hole_init_poses[0][3]
                ),
                "hole_goal": RigidObjectCfg(
                    prim_path = "{ENV_REGEX_NS}/Hole/goal",
                    spawn = sim_utils.CuboidCfg(
                        size=(0.01, 0.01, 0.01),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled = False
                        ), 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True
                        ),
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(self.x_offset, 0, 0.005),
                        rot= (0, 1, 0, 0) 
                    )
                )
            }
        )
        
        self.scene.hole_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            #visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Hole/goal",
                    name="hole_frame",
                    #offset=hole_center_offset
                    offset = OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        self.scene.hole_frame.visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.scene.hole_frame.visualizer_cfg.prim_path="/Visuals/hole_frame"
        

    def get_asset_cfgs(self,
                        radii = (0.015, 0.025),
                        lengths = (0.085, 0.125),
                        clearance = 0.008
    ):
        n = int(np.sqrt(self.scene.num_envs))
        
        raw_lengths = np.linspace(lengths[0], lengths[1], n).tolist()
        raw_radii = np.linspace(radii[0], radii[1], n).tolist()
        
        peg_cfgs = [None for i in range(n*n)]
        hole_cfgs = [None for i in range(n*n)]
        hole_init_states = [None for i in range(n*n)]
        idx = 0
        for l in raw_lengths:
            for r in raw_radii:
                peg_cfgs[idx]= sim_utils.CuboidCfg(
                                    size=(r, r, l),
                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                        diffuse_color=(0.0, 1.0, 0.0), 
                                        metallic=0.2
                                    )
                                )
                
                hole_cfgs[idx], hole_init_states[idx] = self.calcHole(
                                    peg_width = float(1.414 * r + clearance), 
                                    peg_height = float(l)
                                )
                idx += 1
        return peg_cfgs, hole_cfgs, raw_lengths, hole_init_states
    
    def calcHole(self, peg_width, peg_height, diff=1.0):
        """ Returns the offsets and rigidObjectCfgs for a hole 
            peg_width: the actual radius/width of the peg
            peg_height: the actual length/height of the peg
            diff: a difficulty parameter to scale the hole size accordingly
                if diff == 1, then the hole will be peg_width 
        """
        hole_size = peg_width / diff

        w_tot = 0.25

        l = (w_tot - hole_size) / 2.0
        w = w_tot - l

        h = peg_height/2.0
        l_offset = l/2.0 + hole_size/2.0
        w_offset = w/2.0 - hole_size/2.0
        l_pos = [-w_offset, l_offset, w_offset, -l_offset]
        w_pos = [l_offset, w_offset, -l_offset, -w_offset]
        
        init_states= [  
            RigidObjectCfg.InitialStateCfg(
                pos=(
                    l_pos[i] + self.x_offset, 
                    w_pos[i], 
                    h/2.0
                ),
                #rot= [0, 1, 0, 0] #if i in [0, 2] else [0.5, 0.5, -0.5, 0.5]
                rot= (1, 0, 0, 0) if i in [0, 2] else (0.707, 0.0, 0.0, 0.707)
                #rot = [0.0, -0.707, 0.707, 0.0]
            ) for i in range(4)
        ]

        spawn_cfg = sim_utils.CuboidCfg(
            size=(w, l, h),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(), 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0,
                kinematic_enabled=True
            ),
        )
        
        return spawn_cfg, init_states
