import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file = "../assets/quick_bipedal_urdf/urdf/quick_bipedal.urdf",
        pos = [0.0, 0.0, 0.25]
    )
)
scene.build()

jnt_names = [
    "L_thigh_joint", "L_calf_joint",
    "R_thigh_joint", "R_calf_joint",
    "L_wheel_joint", "R_wheel_joint"
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

print(dofs_idx)

robot.set_dofs_kp(
    kp             = np.array([20, 20, 20, 20, 0, 0]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_kv(
    kv             = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_force_range(
    lower          = np.array([-13, -13, -13, -13, -4, -4]),
    upper          = np.array([13, 13, 13, 13, 4, 4]),
    dofs_idx_local = dofs_idx,
)

l_thigh_joint = 1.22

l_calf_joint = -1.92

r_thigh_joint = 1.22

r_calf_joint = -1.92


for i in range(1000):
    
    if i<100:
        robot.set_dofs_position(
            position = np.array([r_thigh_joint, r_calf_joint, l_thigh_joint, l_calf_joint]), 
            dofs_idx_local = dofs_idx[:4],
            zero_velocity = True 
        )
        robot.set_pos([0., 0., 0.25], zero_velocity = False)
        robot.set_quat([1., 0., 0., 0.], zero_velocity = False)
        robot.zero_all_dofs_velocity()

    else:
        robot.control_dofs_position(
            position = np.array([r_thigh_joint, r_calf_joint, l_thigh_joint, l_calf_joint]),
            dofs_idx_local = dofs_idx[0:4]
        )
        robot.control_dofs_velocity(
            velocity = np.array([20, 20]),
            dofs_idx_local = dofs_idx[4:6]
        )
        
    scene.step()

