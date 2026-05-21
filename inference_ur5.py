# %%
import sys
import torch
import numpy as np
import rtde_control, rtde_receive
import pyrealsense2 as rs
import robotiq_gripper
from datetime import datetime
import time
import cv2
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy, PI05Config
from lerobot.policies.pi0 import PI0Policy, PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import Literal

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue

# %%
class Pi0PolicyClient:
    def __init__(self, policy_path, enable_rtc=False):
        self.policy_path = policy_path
        self.device = 'cuda'

        self.action_queue = None

        if enable_rtc:
            policy_cfg = self.setup_rtc()
            model_id = self.policy_path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = PI0Policy.from_pretrained(model_id, policy_cfg=policy_cfg, device=device)

            self.inference_delay = 4
            self.action_queue = ActionQueue(policy_cfg.rtc_config)
        else:
            model_id = self.policy_path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = PI0Policy.from_pretrained(model_id).to(device).eval()

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        self.device = device
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.dataset = None
        self.mean = None
        self.std = None


    def reset(self):
        self.policy.reset()

    def load_training_metadata(self, dataset_path):
        # TODO: Figure out how to load this from postprocessor instead
        dataset = LeRobotDataset(dataset_path)
        self.dataset = dataset
        stats = dataset.meta.stats["actions"]
        mean = np.asarray(stats["mean"][:7])
        std = np.asarray(stats["std"][:7])

        self.mean = mean
        self.std = std

    def get_dataset_episode_frame(self, index):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][index]
        frame = dict(self.dataset[from_idx])

        inference_keys = {
            'observation.images.front',
            'observation.images.wrist',
            'observation.state',
            'task'
        }

        frame = {k: frame[k] for k in inference_keys}

        return frame
    
    def get_dataset_episode_frames(self, index, get_labels=False):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][index]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][index]

        inference_keys = {
            'observation.images.front',
            'observation.images.wrist',
            'observation.state',
            'task'
        }

        actions = []
        frames = []
        for i in range(from_idx, to_idx):
            frame = dict(self.dataset[i])
            if get_labels:
                actions.append(frame["actions"].numpy().tolist())
            frame = {k: frame[k] for k in inference_keys}
            frames.append(frame)

        if get_labels:
            return frames, actions
        else:
            return frames

    def get_action(self, frame, force_normalization=False):
        frame = self.preprocess(frame)
        with torch.inference_mode():
            pred_action = self.policy.select_action(frame)
            pred_action = self.postprocess(pred_action)
            pred_action = pred_action[0][:7].numpy()
        
        if force_normalization:
            pred_action = pred_action * self.std + self.mean

        pred_action = self.normalize_gripper_action(pred_action)
        return pred_action
    
    def setup_rtc(self):
        policy_config = PI05Config()
        policy_config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        return policy_config

    def get_action_chunk(self, frame, force_normalization=False):
        frame = self.preprocess(frame)
        
        prev_actions = self.action_queue.get_left_over()
        actions = self.policy.predict_action_chunk(
          frame,
          inference_delay=self.inference_delay,
          prev_chunk_left_over=prev_actions,
        )

        print(actions)

        self.action_queue.merge(
          actions, actions, self.inference_delay
        )

        pred_action = self.action_queue.get()
        if pred_action == None:
            return np.zeros(7)
        else:
            if force_normalization:
                pred_action = pred_action * self.std + self.mean

            pred_action = self.normalize_gripper_action(pred_action)
            return pred_action

    def normalize_gripper_action(self, action):
        # Round gripper action to 0 or 1
        action[6] = np.round(action[6])
        return action


class Pi05PolicyClient:
    def __init__(self, policy_path, enable_rtc=False):
        self.policy_path = policy_path
        self.device = 'cuda'

        self.action_queue = None

        if enable_rtc:
            policy_cfg = self.setup_rtc()
            model_id = self.policy_path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = PI05Policy.from_pretrained(model_id, policy_cfg=policy_cfg, device=device)

            self.inference_delay = 4
            self.action_queue = ActionQueue(policy_cfg.rtc_config)
        else:
            model_id = self.policy_path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = PI05Policy.from_pretrained(model_id).to(device).eval()

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        self.device = device
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.dataset = None
        self.mean = None
        self.std = None


    def reset(self):
        self.policy.reset()

    def load_training_metadata(self, dataset_path):
        # TODO: Figure out how to load this from postprocessor instead
        dataset = LeRobotDataset(dataset_path)
        self.dataset = dataset
        stats = dataset.meta.stats["actions"]
        mean = np.asarray(stats["mean"][:7])
        std = np.asarray(stats["std"][:7])

        self.mean = mean
        self.std = std

    def get_dataset_episode_frame(self, index):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][index]
        frame = dict(self.dataset[from_idx])

        inference_keys = {
            'observation.images.front',
            'observation.images.wrist',
            'observation.state',
            'task'
        }

        frame = {k: frame[k] for k in inference_keys}

        return frame
    
    def get_dataset_episode_frames(self, index, get_labels=False):
        from_idx = self.dataset.meta.episodes["dataset_from_index"][index]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][index]

        inference_keys = {
            'observation.images.front',
            'observation.images.wrist',
            'observation.state',
            'task'
        }

        actions = []
        frames = []
        for i in range(from_idx, to_idx):
            frame = dict(self.dataset[i])
            if get_labels:
                actions.append(frame["actions"].numpy().tolist())
            frame = {k: frame[k] for k in inference_keys}
            frames.append(frame)

        if get_labels:
            return frames, actions
        else:
            return frames

    def get_action(self, frame, force_normalization=False):
        frame = self.preprocess(frame)
        with torch.inference_mode():
            pred_action = self.policy.select_action(frame)
            pred_action = self.postprocess(pred_action)
            pred_action = pred_action[0][:7].numpy()
        
        if force_normalization:
            pred_action = pred_action * self.std + self.mean

        pred_action = self.normalize_gripper_action(pred_action)
        return pred_action
    
    def setup_rtc(self):
        policy_config = PI05Config()
        policy_config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        return policy_config

    def get_action_chunk(self, frame, force_normalization=False):
        frame = self.preprocess(frame)
        
        prev_actions = self.action_queue.get_left_over()
        actions = self.policy.predict_action_chunk(
          frame,
          inference_delay=self.inference_delay,
          prev_chunk_left_over=prev_actions,
        )

        print(actions)

        self.action_queue.merge(
          actions, actions, self.inference_delay
        )

        pred_action = self.action_queue.get()
        if pred_action == None:
            return np.zeros(7)
        else:
            if force_normalization:
                pred_action = pred_action * self.std + self.mean

            pred_action = self.normalize_gripper_action(pred_action)
            return pred_action

    def normalize_gripper_action(self, action):
        # Round gripper action to 0 or 1
        action[6] = np.round(action[6])
        return action
    


class HardwareSetup():
    def __init__(self, robot_ip):
        # Robot setup
        self.robot_ip = robot_ip
        self.rtde_c, self.rtde_r, self.gripper = self.connect_to_robot()
        self.grip_min_pos = self.gripper.get_min_position()
        self.grip_max_pos = self.gripper.get_max_position()
        print("ROBOT CONNECTED")

        # Camera setup
        self.pipeline1 = None
        self.pipeline2 = None
        self.pipeline3 = None
        self.connect_to_cameras()
        time.sleep(1)
        print("CAMERAS CONNECTED")
        
    def connect_to_robot(self):
        rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        gripper = robotiq_gripper.RobotiqGripper()
        gripper.connect(self.robot_ip, 63352)
        gripper.activate()

        return rtde_c, rtde_r, gripper
    
    def connect_to_cameras(self):
        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            connected_devices.append(detected_camera)

        self.pipeline1 = rs.pipeline()
        config1 = rs.config()
        config1.enable_device(connected_devices[0])
        config1.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipeline1.start(config1)

        self.pipeline2 = rs.pipeline()
        config2 = rs.config()
        config2.enable_device(connected_devices[1])
        config2.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipeline2.start(config2)

        self.pipeline3 = rs.pipeline()
        config3 = rs.config()
        config3.enable_device(connected_devices[2])
        config3.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipeline3.start(config3)
    
    def get_images(self):
        images = []

        frames = self.pipeline1.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.asarray(color_image, dtype=np.uint8)
        images.append(color_image)

        frames = self.pipeline2.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.asarray(color_image, dtype=np.uint8)
        images.append(color_image)

        frames = self.pipeline3.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.asarray(color_image, dtype=np.uint8)
        images.append(color_image)

        return images
    
    def get_robot_state(self):
        joint_state = self.rtde_r.getActualQ()
        gripper_state = self.binarize_gripper_state()
        joint_state.append(gripper_state)
        return joint_state

    def binarize_gripper_state(self):
        # TODO: Check if this makes sense?
        gripper_pos = self.gripper.get_current_position()
        mid_point = (self.grip_max_pos + self.grip_min_pos)/2.0
        if gripper_pos < mid_point:
            return 0.0
        else: 
            return 1.0
        
    def get_observation(self, task):
        images = self.get_images()
        
        state = self.get_robot_state()

        front_image = torch.tensor(images[2], dtype=torch.float32)
        front_image = front_image.permute(2,0,1) / 255.0
        wrist_image = torch.tensor(images[1], dtype=torch.float32)
        wrist_image = wrist_image.permute(2,0,1) / 255.0

        obs = {
            "observation.images.front": front_image,
            "observation.images.wrist": wrist_image,
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "task": task
        }

        return obs
    
    def move_robot_home(self):
        home_q = [np.pi/2.0,
                  -np.pi/1.65,
                  np.pi/1.65,
                  -np.pi/2.0,
                  -np.pi/2.0,
                  np.pi]
        self.move_robot_q(home_q)

    def move_robot_q(self, q):
        self.rtde_c.moveJ(q, 1.05, 1.4)

    def move_robot_eef(self, tcp):
        self.rtde_c.moveL(tcp, 0.15, 0.6)

    def get_robot_q(self):
        return np.array(self.rtde_r.getActualQ())
    
    def get_robot_eef(self):
        return np.array(self.rtde_r.getActualTCPPose())

policy_types = Literal["absq", "abseef", "deltaq", "deltaeef"]

def main(HS, policy, policy_type: policy_types, force_normalization=False):
    ###################################
    ########## CONFIGURATION ##########
    ###################################
    #task = "pull napkin out from the box"
    #task = "pick up napkin"
    #task = "pick up tissue"
    #task = "asduAJKkj dflkasdKJasd hergyt"
    task = "pick up screwdriver and place in box"
    cut_off_time = 45.0
    #policy_type = "deltaeef"
    #policy_type = "abseef"
    zero_state = False
    assert policy_type == "absq" or policy_type == "abseef" or policy_type == "deltaq" or policy_type == "deltaeef", "Policy type must be absq, abseef, deltaq or deltaeef!"
    img_save_path = "/home/pplm/pplm/ICRA/images/pi0_absq_absq/napkin/"    
    save_images = True
    datetime_tag = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    # Initialize environment
    HS.move_robot_home()
    time.sleep(1)
    gripper_open = True
    t0 = time.time()
    j = 0

    while True:
        t1 = time.time()
        if t1 - t0 > cut_off_time:
            break
        # Get observation
        obs = HS.get_observation(task)

        if zero_state:
            state = np.zeros(7)
            obs["observation.state"] = torch.tensor(state, dtype=torch.float32)

        if save_images:
            front_image = obs["observation.images.front"].permute(1,2,0) * 255.0
            front_image = cv2.cvtColor(front_image.numpy(), cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_save_path+f"front_image{j}_{datetime_tag}.png", front_image)
            j += 1

        # Predict action
        action = policy.get_action(obs, force_normalization=force_normalization)
        
        robot_action = action[:6]
        gripper_action = action[6]

        if policy_type == "absq":
            HS.move_robot_q(robot_action)
        
        elif policy_type == "abseef":
            target = robot_action.copy()
            action_rpy = robot_action[3:]
            action_rotmat = R.from_euler('xyz', action_rpy).as_matrix()
            target_rv = R.from_matrix(action_rotmat).as_rotvec()
            target[3:] = target_rv
            HS.move_robot_eef(target)

        elif policy_type == "deltaq":
            current_q = HS.get_robot_q()
            target = current_q + robot_action
            HS.move_robot_q(target)

        elif policy_type == "deltaeef":
            current_eef = HS.get_robot_eef()
            current_rv = current_eef[3:]
            current_rotmat = R.from_rotvec(current_rv).as_matrix()

            action_rpy = robot_action[3:]
            action_rotmat = R.from_euler('xyz', action_rpy).as_matrix()

            target_rotmat = action_rotmat @ current_rotmat
            target_rv = R.from_matrix(target_rotmat).as_rotvec()

            target = current_eef.copy()
            target[:3] += robot_action[:3]
            target[3:] = target_rv

            HS.move_robot_eef(target)



        if gripper_action > 0.0 and gripper_open:
            HS.gripper.move(HS.grip_max_pos, 100, 0)
            gripper_open = False
        elif gripper_action < 1.0 and not gripper_open:
            HS.gripper.move(HS.grip_min_pos, 100, 0)
            gripper_open = True

        if gripper_open:
            HS.gripper.move(HS.grip_min_pos, 100, 0)
        if not gripper_open:
            HS.gripper.move(HS.grip_max_pos, 100, 0)

    HS.move_robot_home()
    HS.gripper.move(HS.grip_min_pos, 100, 0)
    policy.reset()


if __name__ == "__main__":
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_absq_absq_v3/snapshots/72b11e02b3463a15e64ebf1720d622dce68734d7"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--trannguyenle95--ICRA_pi05_abs_joint_abs_joint_v3/snapshots/141848a24bf9c49b2d573343ccfc4cbfb55846ca"
    
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--arturaah--pi0-zero-delta-eef/snapshots/4f2c5fee74a4e3203b4c54d516aa705023c3e26f"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--arturaah--pi05_abs_joint_abs_eef/snapshots/28c3e5be20458094081159177d1027eacc495769"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--arturaah--pi05_abs_joint_delta_eef/snapshots/e6ee34c92590e51f23dc4bea3899bf7a1ade3d08"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--arturaah--pi05_abs_eef_delta_eef_v3/snapshots/7cfee9906b9ae9d682e983614621bda5f46dd19f"

    # ABSQ ABSQ
    #dataset_path = "pmoller/ww_dataset_abs_joint_abs_joint_v3"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_absq_absq_ae_100k/snapshots/8d757dedbdd7f776265e4490d87bdc16161bbb4b"
    #policy_type = "absq"

    # ABSQ DELTAQ
    #dataset_path = "pmoller/ww_dataset_abs_joint_delta_joint_v3"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_absq_dq_ae_100k/snapshots/45bc17883e0521d4105ae9d64fb341f06475e07d"
    #policy_type = "deltaq"

    # ABSQ ABSEEF
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_absq_abseef_ae_100k/snapshots/2a86d23a81a5fa4259a0d378f30ef37fd3d2602d"
    #dataset_path = "pmoller/ww_dataset_abs_joint_abs_eef_v3"
    #policy_type = "abseef"

    # ABSQ DELTAEEF
    #dataset_path = "pmoller/ww_dataset_abs_joint_delta_eef_v3"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_absq_deef_ae_100k/snapshots/024cfb1c1550083790baf0772e8dbeb9dcd3558d"
    #policy_type = "deltaeef"

    # ZERO DELTAEEF
    #dataset_path = "pmoller/ww_dataset_zero_delta_eef_v3"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_zero_deef_ae_100k/snapshots/0d590e89e8ea106c235ae465427e1b33114019da"
    #policy_type = "deltaeef"

    # ABSEEF DELTAEEF
    #dataset_path = "pmoller/ww_dataset_abs_eef_delta_eef_v3"
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi05_abseef_deef_ae_100k/snapshots/f35b496ff6c7b95eb68e5ff3290d52b52410efec"
    #policy_type = "deltaeef"

    # ARTUR ABS JOINT ABS JOINT
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--arturaah--pi05_abs_joint_abs_joint_v3/snapshots/080144a1f63ab2d4cd7e801401bb71b6db587f73"
    #dataset_path = None
    #policy_type = "absq"

    # PI05 BASE
    #dataset_path = None
    #policy_path = "/home/pplm/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
    #policy_type = "abseef"

    # PI0 ABSQ ABSQ
    policy_path = "/home/pplm/.cache/huggingface/hub/models--pmoller--pi0_absq_absq_newversion/snapshots/e959eeca4fd017a22ee6b593bf5914a9011f82f8"
    dataset_path = "pmoller/ww_dataset_abs_joint_abs_joint_v3"
    policy_type = "absq"

    # IT IS A GOOD IDEA TO SET THIS TO TRUE TO CHECK IF YOUR INFERENCE MATCHES A TRAINING DATA SAMPLE BEFORE RUNNING ON HARDWARE
    # ALSO USEFUL TO SEE IF NORMALIZATION IS APPLIED THROUGH THE CONFIG OR NEEDS TO BE FORCED
    dataset_eval = False

    if not dataset_eval:
        if dataset_path == None:
            force_normalization = False
        else:
            force_normalization = True

        # CHANGE ACCORDING TO CHECKPOINT POLICY CLASS BEING USED
        policy = Pi05PolicyClient(policy_path=policy_path)
        #policy = Pi0PolicyClient(policy_path=policy_path)

        if force_normalization:
            policy.load_training_metadata(dataset_path)

        print("Policy succesfully loaded!")
        
        robot_ip = "192.168.1.101"
        HS = HardwareSetup(robot_ip=robot_ip)
        print("Robot connected!")

        time.sleep(1)

        k = 0
        while 1:
            main(HS=HS, 
                 policy=policy, 
                 policy_type=policy_type, 
                 force_normalization=force_normalization)
            k += 1
            print(f"############ TEST {k} FINISHED #############")
            print("Make an input to continue:")
            asd = input()

    else:
        enable_rtc = False
        #policy = Pi05PolicyClient(policy_path=policy_path, enable_rtc=enable_rtc)
        policy = Pi0PolicyClient(policy_path=policy_path)
        policy.load_training_metadata(dataset_path)

        #policy.load_training_metadata("pmoller/ww_dataset_abs_joint_abs_joint_v3")

        episode_index = 5
        frames, labels = policy.get_dataset_episode_frames(episode_index, get_labels=True)

        predictions = []
        if enable_rtc:
            for frame in frames:
                predictions.append(policy.get_action_chunk(frame, force_normalization=True))
        else:
            for frame in frames:
                predictions.append(policy.get_action(frame, force_normalization=True))

        predictions = np.array(predictions)
        labels = np.array(labels)

        pred_j1 = predictions[:,0]
        pred_j2 = predictions[:,1]
        pred_j3 = predictions[:,2]
        pred_j4 = predictions[:,3]
        pred_j5 = predictions[:,4]
        pred_j6 = predictions[:,5]
        pred_grip = predictions[:,6]

        label_j1 = labels[:,0]
        label_j2 = labels[:,1]
        label_j3 = labels[:,2]
        label_j4 = labels[:,3]
        label_j5 = labels[:,4]
        label_j6 = labels[:,5]
        label_grip = labels[:,6]

        fig, ax = plt.subplots(2,3, sharey=True)

        ax[0,0].plot(np.arange(len(pred_j1)), pred_j1, c='r')
        ax[0,0].plot(np.arange(len(label_j1)), label_j1, c='g')

        ax[0,1].plot(np.arange(len(pred_j2)), pred_j2, c='r')
        ax[0,1].plot(np.arange(len(label_j2)), label_j2, c='g')

        ax[0,2].plot(np.arange(len(pred_j3)), pred_j3, c='r')
        ax[0,2].plot(np.arange(len(label_j3)), label_j3, c='g')

        ax[1,0].plot(np.arange(len(pred_j4)), pred_j4, c='r')
        ax[1,0].plot(np.arange(len(label_j4)), label_j4, c='g')

        ax[1,1].plot(np.arange(len(pred_j5)), pred_j5, c='r')
        ax[1,1].plot(np.arange(len(label_j5)), label_j5, c='g')

        ax[1,2].plot(np.arange(len(pred_j6)), pred_j6, c='r')
        ax[1,2].plot(np.arange(len(label_j6)), label_j6, c='g')

        save_name = policy_path.split("hub/")[1]
        save_name = save_name.split("/snapshots")[0]
        save_name = save_name.replace("--", "_")
        save_path = f"/home/pplm/pplm/ICRA/figure_{save_name}_actions.png"

        fig.savefig(save_path)

        
        fig2, ax2 = plt.subplots()

        ax2.plot(np.arange(len(pred_grip)), pred_grip, c='r')
        ax2.plot(np.arange(len(label_grip)), label_grip, c='g')
        save_path = f"/home/pplm/pplm/ICRA/figure_{save_name}_gripper.png"

        fig2.savefig(save_path)

        plt.show()

