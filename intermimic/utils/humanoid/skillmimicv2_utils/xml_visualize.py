import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()

# 配置模拟参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2

# 创建模拟环境
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建一个空的环境
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

# 加载 XML 文件
asset_root = "intermimic/data/assets/parahome/sim_raw_human"
asset_file = "s110.xml"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # 固定模型

try:
    obj_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if obj_asset is None:
        print("Failed to load asset.")
        quit()
except Exception as e:
    print(f"Failed to load asset: {e}")
    quit()

# 设置模型的位置和姿态
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)  # 确保模型在摄像机视野内

# 将模型添加到环境中
obj_handle = gym.create_actor(env, obj_asset, pose, "mocap_parahome", 0, 1)
if obj_handle is None:
    print("Failed to create actor.")
    quit()

# 设置摄像机
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 设置摄像机位置和方向
cam_pos = gymapi.Vec3(3, 3, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 主循环
while not gym.query_viewer_has_closed(viewer):
    # 步进模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新视图
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 处理事件
    gym.poll_viewer_events(viewer)

    # 打印模型的位置和姿态
    state = gym.get_actor_rigid_body_states(env, obj_handle, gymapi.STATE_POS)

# 销毁查看器和模拟
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

