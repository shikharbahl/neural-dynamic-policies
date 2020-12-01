from dnc.envs.base import KMeansEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from gym import error, spaces
import os.path as osp


class PickerPosEnv(KMeansEnv, Serializable):
    """
    Picking a block, where the block position is randomized over a square region

    goal_args is of form ('noisy', center_of_box, half-width of box)

    """


    def __init__(self, goal_args=('noisy', (.6,.2), .1), frame_skip=5, scale=0.05, reward_delay=1, timestep=0.01, *args, **kwargs):
        FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/picker_pos.xml')
        self.goal_args = goal_args
        self.numClose = 0
        self.reward_delay = reward_delay
        self.count = 0
        self.scale = scale
        self.low = np.array([-0.65, -2, -1.4, -1.55, -2.6, -1.8, -0.26, -0.35, -0.24])
        self.high = np.array([0.68,  0.86,  1, 1.75, 1.5,  1.8,  1.5, 1.5, 1.5])
        super(PickerPosEnv, self).__init__(model_path=FILE, frame_skip=frame_skip, *args, **kwargs)
        self.init_qpos[:9] = np.array([-0.07855885, -1.5 ,  0.6731897,  1.5, -2.83926611, 0, 0.2, 0.2, 0.2])
        self.sim.model.opt.timestep = timestep
        Serializable.__init__(self, goal_args, frame_skip, *args, **kwargs)



    def _get_obs(self):
        return self.get_current_obs()

    def get_current_obs(self):
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        return np.concatenate([
            self.sim.data.qpos.flat[:],
            self.sim.data.qvel.flat[:],
            finger_com,
        ]).reshape(-1)

    def step(self,action):
        if len(action) < 9:
            action = np.zeros(9)
            return Step(self.get_current_obs(), 0, False)

        action[-3:] = action[-2]

        action *= self.scale
        qpos = self.sim.data.qpos.copy()
        qp = qpos[:9]
        qp += action
        qp = np.clip(qp, self.low, self.high)
        qpos[:9] = qp
        self.sim.data.qpos[:9] = qp
        self.sim.forward()
        self.sim.data.ctrl[:] = self.sim.data.qfrc_bias[:9]

        reward = 0
        timesInHand = 0

        for _ in range(self.frame_skip):
            self.sim.step()
            step_reward = self.reward()
            timesInHand += step_reward > 0
            reward += step_reward


        self.count += 1

        done = reward == 0 and self.numClose > 0 # Stop it if the block is flinged

        timesInHand = 0
        ob = self.get_current_obs()

        new_com = self.sim.data.subtree_com[0]
        self.current_com = new_com
        self.dcom = new_com - self.current_com

        obj_position = self.get_body_com("object")
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.

        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        is_success = False
        if dist_1 < .15 and obj_position[2] > 0.08:
            is_success = True

        return Step(ob, float(reward), False, distance=dist_1, timeInHand=timesInHand, success=is_success)

    def reward(self):
        if self.count % self.reward_delay == 0:
            self.obj_position = self.get_body_com("object").copy()
        obj_position = self.obj_position

        if obj_position[2] < 0.08:
            return 0

        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.

        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        if dist_1 < .1:
            self.numClose += 1
            return obj_position[2]
        else:
            return 0

    def sample_position(self,goal_type,center=(0.6,0.2),noise=0):
        if goal_type == 'fixed':
            return [center[0],center[1],.03]
        elif goal_type == 'noisy':
            x,y = center
            return [x+(np.random.rand()-0.5)*2*noise,y+(np.random.rand()-0.5)*2*noise,.03]
        else:
            raise NotImplementedError()

    def retrieve_centers(self,full_states):
        return full_states[:,9:12]

    def propose_original(self):
        return self.sample_position(*self.goal_args)

    @overrides
    def reset(self):
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1)
        qpos[1] = -1
        noise = np.random.uniform(-0.02, 0.02, 3)
        noise[-1] = 0
        self.position = np.array([0.55, 0.15, 0.03]) + noise
        qpos[9:12] = self.position
        qvel[9:12] = 0

        self.set_state(qpos.reshape(-1), qvel)

        self.numClose = 0

        self.current_com = self.sim.data.subtree_com[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = +0.0
        self.viewer.cam.elevation = -40

    @overrides
    def log_diagnostics(self, paths, prefix=''):

        timeOffGround = np.array([
            np.sum(path['env_infos']['timeInHand'])*.01
        for path in paths])

        timeInAir = timeOffGround[timeOffGround.nonzero()]

        if len(timeInAir) == 0:
            timeInAir = [0]

        avgPct = lambda x: round(np.mean(x) * 100, 2)

        logger.record_tabular(prefix+'PctPicked', avgPct(timeOffGround > .3))
        logger.record_tabular(prefix+'PctReceivedReward', avgPct(timeOffGround > 0))

        logger.record_tabular(prefix+'AverageTimeInAir',np.mean(timeOffGround))
        logger.record_tabular(prefix+'MaxTimeInAir',np.max(timeOffGround ))
