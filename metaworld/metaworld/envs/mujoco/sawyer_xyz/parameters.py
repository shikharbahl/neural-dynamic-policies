import numpy as np

push_params = dict(
    fixed_goal_unconstrainted = dict(
            random_init=False,
            goal_low=(-0.1, 0.8, 0.05),
            goal_high=(0.1, 0.9, 0.3),
            hand_low = (-0.5, 0.40, 0.05),
            hand_high = (0.5, 1, 0.5),
            init_config = {
                'obj_init_angle': .3,
                'obj_init_pos': np.array([0, 0.6, 0.02]),
                'hand_init_pos': np.array([0, .6, .2]),
            }
    ),
    random_goal_unconstrained=dict(
            random_init=True,
            goal_low=(-0.1, 0.8, 0.05),
            goal_high=(0.1, 0.9, 0.3),
            hand_low = (-0.5, 0.40, 0.05),
            hand_high = (0.5, 1, 0.5),
            init_config = {
                'obj_init_angle': .3,
                'obj_init_pos': np.array([0, 0.6, 0.02]),
                'hand_init_pos': np.array([0, .6, .2]),
            }

    ),
    random_goal_constrained=dict(
            random_init=True,
            goal_low=(-0.1, 0.8, 0.05),
            goal_high=(0.1, 0.9, 0.3),
            hand_low = (-0.5, 0.40, 0.05),
            hand_high = (0.5, 1, 0.25),
            init_config = {
                'obj_init_angle': .3,
                'obj_init_pos': np.array([0, 0.6, 0.02]),
                'hand_init_pos': np.array([0, .6, .2]),
            }
    ),
)


soccer_params = dict(
    fixed_goal_unconstrainted = dict(
        random_init=False,
        goal_low = (-0.1, 0.8, 0.03),
        goal_high = (0.1, 0.9, 0.03),
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.03]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
    ),
    random_goal_unconstrained=dict(
        random_init=True,
        goal_low = (-0.1, 0.8, 0.03),
        goal_high = (0.1, 0.9, 0.03),
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.03]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
    ),
    random_goal_constrained=dict(
        random_init=True,
        goal_low = (-0.1, 0.8, 0.03),
        goal_high = (0.1, 0.9, 0.03),
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.2),
        init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.03]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
    ),
)

door_params = dict(
    fixed_goal_unconstrainted = dict(
        random_init=False,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
    random_goal_unconstrained=dict(
        random_init=True,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
    random_goal_constrained=dict(
        random_init=True,
        hand_low = (-0.4, 0.40, 0.05),
        hand_high = (0.4, 1, 0.25),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    )
)


drawer_params = dict(
    fixed_goal_unconstrainted = dict(
        random_init=False,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
    random_goal_unconstrained=dict(
        random_init=True,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
    random_goal_constrained=dict(
        random_init=True,
        hand_low = (-0.25, 0.40, 0.05),
        hand_high = (0.25, 1, 0.25),
        init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
)
place_params = dict(
    fixed_goal_unconstrainted = dict(
        random_init=False,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        goal_low = (-0.1, 0.8, 0.001),
        goal_high = (0.1, 0.9, 0.001),
        init_config = {
            'obj_init_pos':np.array([0, 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),

    random_goal_unconstrained=dict(
        random_init=True,
        hand_low = (-0.5, 0.40, 0.05),
        hand_high = (0.5, 1, 0.5),
        goal_low = (-0.1, 0.8, 0.001),
        goal_high = (0.1, 0.9, 0.001),
        init_config = {
            'obj_init_pos':np.array([0, 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
    ),
    random_goal_constrained=dict(
        random_init=True,
        hand_low = (-0.25, 0.50, 0.05),
        hand_high = (0.25, 1, 0.5),
        goal_low = (-0.1, 0.8, 0.001),
        goal_high = (0.1, 0.9, 0.001),
        init_config = {
            'obj_init_pos':np.array([0, 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }

    )
)
faucet_params = dict(
    fixed_goal_unconstrainted = dict(
        random_init=False,
        hand_low = (-0.5, 0.40, -0.15),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.05]),
            'hand_init_pos': np.array([0., .6, .2]),
        }
    ),
    random_goal_unconstrained=dict(
        random_init=True,
        hand_low = (-0.5, 0.40, 0.04),
        hand_high = (0.5, 1, 0.5),
        init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.05]),
            'hand_init_pos': np.array([0., .6, .2]),
        }
    ),
    random_goal_constrained=dict(
        random_init=True,
        hand_low = (-0.35, 0.40, 0.04),
        hand_high = (0.35, 1, 0.35),
        init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.05]),
            'hand_init_pos': np.array([0., .6, .2]),
        }
    )
)
