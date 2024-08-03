import sys
import yaml
import os, glob
from data_utils import objects_filter
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from export_utils import *
import random
import queue

 # 获取心跳一致的data
def _retrieve_data(q, frame):
    while True:
            data = q.get()
            if data.frame == frame:
                return data
# 获取相机内参
def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k

# 我觉得是设置一个距离，因为车辆自身传感器感知范围有限，不可能是在那都能获取到所有actor信息，所以根据车辆自身定位和其他环境之间的距离，判断是否可以感知到
def filter_by_distance(data_dict, dis):
    environment_objects = data_dict["environment_objects"]
    actors = data_dict["actors"]
    for agent,_ in data_dict["agents_data"].items():
        data_dict["environment_objects"] = [obj for obj in environment_objects if
                                            distance_between_locations(obj.transform.location, agent.get_location())
                                            < dis]
        data_dict["actors"] = [act for act in actors if
                                            distance_between_locations(act.get_location(), agent.get_location())<dis]

# 就是求距离的公式,勾股定理 c = sqrt(a * a + b * b)
def distance_between_locations(location1, location2):
    return math.sqrt(pow(location1.x-location2.x, 2)+pow(location1.y-location2.y, 2))

def main():
    # config配置，各种传感器和carla配置
    with open("configs.yaml", 'r') as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        except:
            cfg = yaml.load(f)


    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager()
    init_settings = None
    frame = None
    actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
    data = {"sensor_data": {}, "environment_data": None}  # 记录每一帧的数据
    vehicle = None

#--------------------------------------------构建文件路径------------------------------------------------#
    PHASE = "training"
    root_path = cfg["SAVE_CONFIG"]["ROOT_PATH"]
    OUTPUT_FOLDER = os.path.join(root_path, PHASE)
    # 创建文件存放目录
    folders = ['calib', 'image', 'kitti_label', 'carla_label', 'velodyne']

    for folder in folders:
        directory = os.path.join(OUTPUT_FOLDER, folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
    KITTI_LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'kitti_label/{0:06}.txt')
    CARLA_LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'carla_label/{0:06}.txt')
    IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image/{0:06}.png')
    CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
    """获取文件夹中存在的数据量, 判断是否覆盖还是append"""
    label_path = os.path.join(OUTPUT_FOLDER, 'kitti_label/')
    num_existing_data_files = len(
        [name for name in os.listdir(label_path) if name.endswith('.txt')])
    print("当前存在{}个数据".format(num_existing_data_files))
    if num_existing_data_files == 0:
        return 0
    answer = input(
        "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
            OUTPUT_FOLDER))
    if answer.upper() == "O":
        logging.info(
            "Resetting frame number to 0 and overwriting existing")
        return 0
    logging.info("Continuing recording data on frame number {}".format(
        num_existing_data_files))
    
    # 获取目前存在的文件数量
    captured_frame_no = num_existing_data_files
    try:
        # carla产生actor以及传感器舰艇
        # 获取初始化的carla世界设置，以及后续要用的设置
        init_settings = world.get_settings()
        settings = world.get_settings()
        # 设置同步模式，这样carla可以自己控制，不会运行的太快
        settings.synchronous_mode = True
        # 每帧图片之间设定步长
        settings.fixed_delta_seconds = 0.05
        # carla世界使用该setting
        world.apply_settings(settings)
        
#-------------------------------------------- 生成车辆和行人------------------------------------------------#
        # 获取文件中设置的车辆和行人的数量
        num_of_vehicles = cfg["CARLA_CONFIG"]["NUM_OF_VEHICLES"]
        num_of_walkers = cfg["CARLA_CONFIG"]["NUM_OF_WALKERS"]

        # 生成车辆actors
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # 判断出生点的数量，防止设置的车辆数目大于生成点的数目
        if num_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
            num_of_vehicles = num_of_vehicles
        elif num_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, number_of_spawn_points)
            num_of_vehicles = number_of_spawn_points

        batch = []
        for n, transform in enumerate(spawn_points):
            # 设置车辆的颜色(color)和transform(出生位置)
            if n >= num_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            # 获取生成车辆颜色和对应位置的指令
            batch.append(carla.command.SpawnActor(blueprint, transform))

            # 批量执行batch中的指令
            for response in client.apply_batch_sync(batch):
                if response.error:
                    continue
                else:
                    # 因为还没有设置智能体(agent)，所以先放在非智能体部分(non_agents)
                    actors["non_agents"].append(response.actor_id)

        # 生成行人actors
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points = []
        # 设置出生点
        for i in range(num_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 批量执行生成指令
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                # walkers中添加actor_id号码
                actors["walkers"].append(response.actor_id)
        print("spawn {} vehicles and {} walkers".format(len(actors["non_agents"]),
                                                        len(actors["walkers"])))
        world.tick()


#-------------------------------------------- 设置运动路线------------------------------------------------#
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        # 获得车辆
        vehicle_actors = world.get_actors(actors["non_agents"])
        # 设置为自动驾驶模式
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # 给每个行人设置ai属性
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for i in range(len(actors["walkers"])):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
                                                  actors["walkers"][i]))
        controllers_id = []
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        # 设置可以在道路上行走或在道路上的任何点穿过的行人的百分比。值应介于0.0和1.0之间。
        # 例如，值为0.1将允许10%的行人在道路上行走。默认值为0.0。
        world.set_pedestrians_cross_factor(0.2)

        for con_id in controllers_id:
            # start walker
            world.get_actor(con_id).start()
            # set walk to random point
            destination = world.get_random_location_from_navigation()
            world.get_actor(con_id).go_to_location(destination)
            # max speed
            world.get_actor(con_id).set_max_speed(10)

#-------------------------------------------- 设置自己的车辆和相对应的传感器------------------------------------------------#
        vehicle_bp = random.choice(world.get_blueprint_library().filter(cfg["AGENT_CONFIG"]["BLUEPRINT"]))
        trans_cfg = cfg["AGENT_CONFIG"]["TRANSFORM"]
        # 设置车辆的定位和旋转角度
        transform = carla.Transform(carla.Location(trans_cfg["location"][0],
                                               trans_cfg["location"][1],
                                               trans_cfg["location"][2]),
                                carla.Rotation(trans_cfg["rotation"][0],
                                               trans_cfg["rotation"][1],
                                               trans_cfg["rotation"][2]))
        # 同样是设置车辆的位置和自动驾驶模式，然后将该车辆放入actors的agents中
        transform = random.choice(world.get_map().get_spawn_points())
        agent = world.spawn_actor(vehicle_bp, transform)
        agent.set_autopilot(True, traffic_manager.get_port())
        actors["agents"].append(agent)

        actors["sensors"][agent] = []
        for sensor, config in cfg["SENSOR_CONFIG"].items():
            sensor_bp = world.get_blueprint_library().find(config["BLUEPRINT"])
            for attr, val in config["ATTRIBUTE"].items():
                sensor_bp.set_attribute(attr, str(val))
            trans_cfg = config["TRANSFORM"]
            transform = carla.Transform(carla.Location(trans_cfg["location"][0],
                                                       trans_cfg["location"][1],
                                                       trans_cfg["location"][2]),
                                        carla.Rotation(trans_cfg["rotation"][0],
                                                       trans_cfg["rotation"][1],
                                                       trans_cfg["rotation"][2]))
            sensor = world.spawn_actor(sensor_bp, transform, attach_to=agent)
            actors["sensors"][agent].append(sensor)
        world.tick()
        # 开启sensor的数据采集
        for agent, sensors in actors["sensors"].items():
            data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

        step = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        while True:
            if step % STEP ==0:
                ret = {"environment_objects": None, "actors": None, "agents_data": {}}
                frame = world.tick()
                # 枚举声明，其中包含可用于过滤carla.World.get_level_bbs（）返回的边界框的不同标记。这些值对应于场景中元素所具有的语义标记
                ret["environment_objects"] = world.get_environment_objects(carla.CityObjectLabel.Any)
                # 获取所有的actors
                ret["actors"] = world.get_actors()
                # 图像的分辨率
                image_width = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
                image_height = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]
                for agent, dataQue in data["sensor_data"].items():
                    data = [_retrieve_data(q, frame) for q in dataQue]
                    assert all(x.frame == frame for x in data)
                    ret["agents_data"][agent] = {}
                    ret["agents_data"][agent]["sensor_data"] = data
                    ret["agents_data"][agent]["intrinsic"] = camera_intrinsic(image_width, image_height)
                    ret["agents_data"][agent]["extrinsic"] = np.asmatrix(
                        actors["sensors"][agent][0].get_transform().get_matrix())
                # 根据距离过滤一些车辆感知不到的标签
                filter_by_distance(ret, cfg["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"])

                # 获取到所有actors、sensor以及actors的标签
                data = ret
                data = objects_filter(data)
                lidar_fname = LIDAR_PATH.format(captured_frame_no)
                kitti_label_fname = KITTI_LABEL_PATH.format(captured_frame_no)
                carla_label_fname = CARLA_LABEL_PATH.format(captured_frame_no)
                img_fname = IMAGE_PATH.format(captured_frame_no)
                calib_filename = CALIBRATION_PATH.format(captured_frame_no)

                for agent, dt in data["agents_data"].items():

                    camera_transform= carla.Transform(carla.Location(cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["location"][0],
                                               cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["location"][1],
                                               cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["location"][2]),
                                carla.Rotation(cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["rotation"][0],
                                               cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["rotation"][1],
                                               cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]["rotation"][2]))
                    
                    lidar_transform = carla.Transform(carla.Location(cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["location"][0],
                                               cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["location"][1],
                                               cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["location"][2]),
                                carla.Rotation(cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["rotation"][0],
                                               cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["rotation"][1],
                                               cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"]["rotation"][2]))

                    save_ref_files(OUTPUT_FOLDER, captured_frame_no)
                    save_image_data(img_fname, dt["sensor_data"][0])
                    save_label_data(kitti_label_fname, dt["kitti_datapoints"])
                    save_label_data(carla_label_fname, dt['carla_datapoints'])
                    save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
                    save_lidar_data(lidar_fname, dt["sensor_data"][2])
                captured_frame_no += 1
                print(step / STEP)
            else:
                world.tick()
            step+=1
    finally:
        # 销毁车辆
        for agent in actors["agents"]:
            for sensor in actors["sensors"][agent]:
                sensor.destroy()
            agent.destroy()
        # 销毁actor和walker
        batch = []
        for actor_id in actors["non_agents"]:
            batch.append(carla.command.DestroyActor(actor_id))
        for walker_id in actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        client.apply_batch_sync(batch)
        world.apply_settings(init_settings)



if __name__ == '__main__':
    main()