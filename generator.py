from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from data_utils import objects_filter
import carla
def main():
    # config配置，各种传感器和carla配置
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg)
    # 设置数据集路径，保存路径，存储路径
    dtsave = DataSave(cfg)
    try:
        # carla产生actor以及传感器舰艇
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()
        step = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        while True:
            if step % STEP ==0:
                data = model.tick()
                data = objects_filter(data)
                dtsave.save_training_files(data)
                print(step / STEP)
            else:
                model.world.tick()
            
            step+=1
    finally:
        model.setting_recover()


if __name__ == '__main__':
    main()
