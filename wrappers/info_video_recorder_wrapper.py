"""Wrapper for recording videos. That uses the rgb image in the info observation"""
import os
from typing import Callable, Optional

import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.monitoring import video_recorder

from wrappers.video_recoder_wrapper import capped_cubic_video_schedule
import torch

import cv2
import torchvision
import torchvision.utils as vutils

class InfoRecordVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
        num_envs: int = 8
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable moviepy logger or not.
        """
        gym.Wrapper.__init__(self, env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        #self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.video_name = None
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False
        
        self.agent = None
        self.out_img = None
        self.num_recorded_envs = num_envs
        self.frames = None
    
    def is_recording(self):
        return self.recording

    def set_agent(self, agent):
        self.agent = agent

    def set_video_name(self, name: str):
        self.video_name = name

    def capture_frame(self, obs, infos):
        """Concatanates images, and stores them until ready to save"""
        imgs = obs['info']['img']
        num_envs, w, h, c = imgs.size()
        if self.out_img is None:
            self.num_recorded_envs = min(num_envs, self.num_recorded_envs)
            self.out_img = torch.zeros(
                (2 * w, self.num_recorded_envs//2 * h, c),
                dtype=imgs.dtype,
                device=imgs.device
            )
            self.frames = torch.zeros(
                (self.video_length, 2 * w, self.num_recorded_envs//2 * h, c),
                dtype=torch.uint8,
                device=imgs.device
            )

        # would like this not to be a for loop but couldn't figure it out
        # should not do this for every env, but rather a subset
        for i in range(self.num_recorded_envs):
            y = i // 2 #(self.num_recorded_envs//2)
            x = i % 2 #(self.num_recorded_envs//2)
            #print(f"i:{i}; \tx:{x*w}-{(x+1)*w} \ty:{y*h}-{(y+1)*h}")
            self.out_img[x*w:(x+1)*w, y*h:(y+1)*h, :] = imgs[i,:,:,:]
        
        #cpu_tiled = self.out_img.cpu().detach().numpy()
        #print("about to call cv")
        #cv2.imshow('Tiled Images', cpu_tiled)
        #cv2.waitKey(1)
        #print("complete cv")
        #print(self.recorded_frames, self.frames.size(), self.out_img.size())
        #self.out_img[:,:,[0,2]] = self.out_img[:, :, [2,0]]
        #print(self.frames.size(), self.out_img.size())
        self.frames[self.recorded_frames,:,:,:] = (1 - self.out_img) * 255
        self.recorded_frames += 1


    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations, info = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            #assert self.video_recorder is not None
            #self.video_recorder.recorded_frames = []
            #self.video_recorder.capture_frame()
            self.capture_frame(observations, info)
            if self.video_length > 0:
                if self.recorded_frames >= self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder(observations, info)
        return observations, info

    def start_video_recorder(self, obs, info):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        if self.video_name is not None:
            video_name = self.video_name
            if "STEP_NUM" in video_name:
                video_name = video_name.replace("STEP_NUM", f"{self.step_id}")

        #print(self.video_folder, "\n", video_name)
        base_path = os.path.join(self.video_folder, video_name)
        self.video_path_full = base_path + ".mp4"
        #print(base_path)
        #self.video_recorder = video_recorder.VideoRecorder(
        #    env=self.env,
        #    base_path=base_path,
        #    metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        #    disable_logger=self.disable_logger,
        #)

        if self.agent is not None and "eval" in base_path:
            self.agent.track_video_path("Eval Videos", base_path + ".mp4", self.step_id)
        elif self.agent is not None and "train" in base_path:
            self.agent.track_video_path("Training Videos", base_path + ".mp4", self.step_id)

        #self.video_recorder.capture_frame()
        self.capture_frame(obs, info)
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            if "train" in self.video_name:
                self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                #assert self.video_recorder is not None
                #self.video_recorder.capture_frame()
                self.capture_frame(observations, infos)
                if self.video_length > 0:
                    if self.recorded_frames >= self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder(observations, infos)


        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        #if self.recording:
        #    assert self.video_recorder is not None
        #    self.video_recorder.close()
        if self.frames is not None and self.recorded_frames > 25:
            #print(self.frames[:self.recorded_frames,:,:,:].cpu().size())
            torchvision.io.write_video(
                self.video_path_full, 
                self.frames[:self.recorded_frames,:,:,:].cpu(), 
                fps=10
            )
        self.recording = False
        self.recorded_frames = 0

    """
    def render(self, *args, **kwargs):
        #Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs.
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)
    """

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        #self.close_video_recorder()