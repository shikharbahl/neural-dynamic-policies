"""

This file is from multiworld: https://github.com/vitchyr/multiworld

Based on rllab's serializable.py file, and multiworlds mujoco_env.py

https://github.com/rll/rllab
"""

import inspect
import sys
import os
from sklearn.cluster import KMeans
import numpy as np
from gym import error, spaces
from gym.utils import seeding
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class MjEnv(gym.Env):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """
    def __init__(self, model_path, frame_skip, device_id=-1, automatically_set_spaces=True):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        if device_id == -1 and 'gpu_id' in os.environ:
            device_id =int(os.environ['gpu_id'])
        self.device_id = device_id
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        if automatically_set_spaces:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

            high = np.inf*np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            # width, height = 500, 500
            width, height = 1750, 1000
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def initialize_camera(self, init_fctn):
        sim = self.sim
        viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=self.device_id)
        # viewer = mujoco_py.MjViewer(sim)
        init_fctn(viewer.cam)

        sim.add_render_context(viewer)



class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


class MujocoEnv(MjEnv):
    def __init__(self, frame_skip=1, *args, **kwargs):
        self.bd_index = None
        # import ipdb; ipdb.set_trace()
        self.geom_names_to_indices = None
        super().__init__(frame_skip=frame_skip, *args, **kwargs)
        self.frame_skip = frame_skip
        self.geom_names_to_indices = {name:index for index,name in enumerate(self.model.geom_names)}


    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        # self.model._compute_subtree()  # pylint: disable=W0212
        self.sim.forward()

    def get_body_com(self, body_name):
        # Speeds up getting body positions

        if self.bd_index is None:
            self.bd_index = {name:index for index,name in enumerate(self.model.body_names)}
        idx = self.bd_index[body_name]
        return self.sim.data.subtree_com[idx]

    def touching(self, geom1_name, geom2_name):
        if not self.geom_names_to_indices:
            return False
        idx1 = self.geom_names_to_indices[geom1_name]
        idx2 = self.geom_names_to_indices[geom2_name]
        for c in self.sim.data.contact:
            if (c.geom1 == idx1 and c.geom2 == idx2) or (c.geom1 == idx2 and c.geom2 == idx1):
                return True
        return False

    def touching_group(self, geom1_name, geom2_names):
        if not self.geom_names_to_indices:
            return False
        idx1 = self.geom_names_to_indices[geom1_name]
        idx2s = set([self.geom_names_to_indices[geom2_name] for geom2_name in geom2_names])

        for c in self.sim.data.contact:
            if (c.geom1 == idx1 and c.geom2 in idx2s) or (c.geom1 in idx2s and c.geom2 == idx1):
                return True
        return False

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def get_viewer(self):
        if self.viewer is None:
            viewer = super().get_viewer()
            self.viewer_setup()
            return viewer
        else:
            return self.viewer


class KMeansEnv(MujocoEnv):
    def __init__(self,kmeans_args=None,*args,**kwargs):
        if kmeans_args is None:
            self.kmeans = False
        else:
            self.kmeans = True
            self.kmeans_centers = kmeans_args['centers']
            self.kmeans_index = kmeans_args['index']

        super(KMeansEnv, self).__init__(*args, **kwargs)

    def propose_original(self):
        raise NotImplementedError()

    def propose_kmeans(self):
        while True:
            proposal = self.propose_original()
            distances = np.linalg.norm(self.kmeans_centers-proposal,axis=1)
            if np.argmin(distances) == self.kmeans_index:
                return proposal

    def propose(self):
        if self.kmeans:
            return self.propose_kmeans()
        else:
            return self.propose_original()

    def create_partitions(self,n=10000,k=3):
        X = np.array([self.reset() for i in range(n)])
        kmeans = KMeans(n_clusters=k).fit(X)
        return self.retrieve_centers(kmeans.cluster_centers_)

    def retrieve_centers(self,full_states):
        raise NotImplementedError()

    def get_param_values(self):
        if self.kmeans:
            return dict(kmeans=True, centers=self.kmeans_centers, index=self.kmeans_index)
        else:
            return dict(kmeans=False)

    def set_param_values(self, params):
        self.kmeans = params['kmeans']
        if self.kmeans:
            self.kmeans_centers = params['centers']
            self.kmeans_index = params['index']


def create_env_partitions(env, k=4):

    assert isinstance(env, KMeansEnv)
    cluster_centers = env.create_partitions(k=k)

    envs = [env.clone(env) for i in range(k)]
    for i,local_env in enumerate(envs):
        local_env.kmeans = True
        local_env.kmeans_centers = cluster_centers
        local_env.kmeans_index = i

    return envs
