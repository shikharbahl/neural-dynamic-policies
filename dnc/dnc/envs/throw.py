from dnc.envs.base  import KMeansEnv
import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from gym import error, spaces

import os.path as osp
from dnc import envs as dnc_envs



class ThrowerPosEnv(KMeansEnv, Serializable):



    def __init__(self, box_center=(0,0), box_noise=0, frame_skip=5, scale=0.05, reward_delay=1, timestep=0.01, *args, **kwargs):
        FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/throw_pos.xml')
        self.box_center = box_center
        self.box_noise = box_noise
        self.relativeBoxPosition = np.zeros(3)
        self.init_block_goal_dist = 1.0
        self.scale = scale
        self.reward_delay = reward_delay
        self.count = 0
        self.low = np.array([-0.65, -1.7, -1.4, -1.8, -2.6, -1.8, -0.26, -0.35, -0.24])
        self.high = np.array([0.68,  1.5,  1,     1.8, 1.5,  1.8,  1.2, 1, 1])
        super(ThrowerPosEnv, self).__init__(model_path = FILE, frame_skip=frame_skip, *args, **kwargs)
        self.sim.model.opt.timestep = timestep
        Serializable.__init__(self, box_center, box_noise, frame_skip, *args, **kwargs)
        self.init_qpos[:9] = np.array([-0.07855885, -1.5 ,  0.6731897,  1.5, -2.83926611, 0, 0.9, 0.9, 0.9])

        self.action_space = spaces.Box(low=-np.ones(9)*5, high=np.ones(9)*5, dtype=np.float32)

    def _get_obs(self):
        return self.get_current_obs()

    def get_current_obs(self):
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        return np.concatenate([
            self.sim.data.qpos.flat[:],
            self.sim.data.qvel.flat[:],
            finger_com,
            self.relativeBoxPosition,
        ]).reshape(-1)

    def step(self, action):
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
        for _ in range(self.frame_skip):
            self.sim.step()

        done = False
        onGround = self.touching_group("geom_object", ["ground", "goal_wall1", "goal_wall2", "goal_wall3"])

        reward += self.final_reward()
        self.count += 1


        ob = self.get_current_obs()
        new_com = self.current_com = self.sim.data.subtree_com[0]
        self.dcom = new_com - self.current_com
        #
        # # Recording Metrics
        #
        in_success = False
        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")
        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")
        distance = np.linalg.norm((goal_position - obj_position)[:2])
        if distance < 0.19:
            is_success = True
        normalizedDistance = distance / self.init_block_goal_dist

        return Step(ob, float(reward), done, distance=distance, norm_distance=normalizedDistance, success=in_success)

    @overrides
    def reset(self):
        self.numClose = 0
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1)

        qpos[1] = -.1

        self.set_state(qpos.reshape(-1), qvel)
        hand_pos = self.get_body_com('jaco_link_hand')
        noise = np.random.uniform(-0.02, 0.02, 3)
        noise[-1] = 0

        qpos[9:12] = np.array((0.6, 0.1,0.03)) + noise
        qvel[9:12] = 0
        self.set_state(qpos.reshape(-1), qvel)
        # Save initial distance between object and goal
        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")
        self.relativeBoxPosition = goal_position - obj_position
        self.init_block_goal_dist = np.linalg.norm(obj_position - goal_position)

        self.current_com = self.sim.data.subtree_com[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def timestep_reward(self):
        obj_position = self.get_body_com("object")

        if obj_position[2] < 0.08:
            return 0

        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.

        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        if dist_1 < .15 and obj_position[0] > .2:
            self.numClose += 1
            return obj_position[2]
        else:
            return 0

    def final_reward(self):
        if self.count % self.reward_delay == 0:
            self.obj_position = self.get_body_com("object").copy()
        obj_position = self.obj_position
        goal_position = self.get_body_com("goal")

        vec_2 = obj_position - goal_position
        dist_2 = np.linalg.norm(vec_2[:2])
        normalized_dist_2 = dist_2 / self.init_block_goal_dist
        clipped_dist_2 = min(1.0, normalized_dist_2)

        if dist_2 < .18:
            return 40

        reward = 1 - clipped_dist_2

        return 40 * reward



    def retrieve_centers(self,full_states):
        return full_states[:,16:18]-self.init_qpos.copy().reshape(-1)[-2:]

    def propose_original(self):
        return np.array(self.box_center) + 2*(np.random.random(2)-0.5)*self.box_noise

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = +60.0
        self.viewer.cam.elevation = -30

    @overrides
    def log_diagnostics(self, paths, prefix=''):

        progs = np.array([
            path['env_infos']['norm_distance'][-1] for path in paths
        ])

        inGoal = np.array([
            path['env_infos']['distance'][-1] < .1 for path in paths
        ])

        avgPct = lambda x: round(np.mean(x)*100,3)

        logger.record_tabular(prefix+'PctInGoal', avgPct(inGoal))
        logger.record_tabular(prefix+'AverageFinalDistance', np.mean(progs))
        logger.record_tabular(prefix+'MinFinalDistance', np.min(progs ))
