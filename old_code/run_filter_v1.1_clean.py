import os
import time
import tqdm
import signal
import shutil
import subprocess as sp
import dbus
sys_bus = dbus.SystemBus()
num_rounds = 1
import os, psutil

fusion = ["only uwb", "uwb and vision"]
print("------ START ------")

tri_flag = True
# bag_file = "cali_4robots_data_02"
bag_file = "cali_4robots_data_04"
# bag_file = "4robots_data_07"
for i in [0,1]:
    print(f"------{fusion[i]}------")
    for r in tqdm.tqdm(range(num_rounds)):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221013/cali_4robots_data_04"], stdout=sp.PIPE, stderr=sp.STDOUT)
        time.sleep(0.1)
        p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots_v4.1_clean.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

        p2 = sp.Popen(["python", "triangulation_ros2_uwb_position_v2.0.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        if os.path.exists("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)):
            shutil.rmtree("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)) 
        # time.sleep(100)
        time.sleep(150)
        p3 = sp.Popen(["ros2", "bag", "record",  "/tri_turtle01_pose", "/tri_turtle03_pose", "/tri_turtle04_pose", "/real_turtle01_pose", "/real_turtle03_pose",
                 "/real_turtle04_pose", "/real_turtle05_pose", "/pf_turtle01_pose","/pf_turtle03_pose", "/pf_turtle04_pose", "-o", "./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)], stdout=sp.PIPE, stderr=sp.STDOUT)

        # print(psutil.Process(p3).memory_info().rss / 1024 ** 2)
        p0.wait()
        p1.send_signal(signal.SIGINT)
        p2.send_signal(signal.SIGINT)
        p3.send_signal(signal.SIGINT)
        time.sleep(1)

# ck_srv = sys_bus.get_object('org.freedesktop.ConsoleKit',
#                                 '/org/freedesktop/ConsoleKit/Manager')
# ck_iface = dbus.Interface(ck_srv, 'org.freedesktop.ConsoleKit.Manager')
# stop_method = ck_iface.get_dbus_method("Stop")
# stop_method()

# os.system("shutdown /s /t 1")
print("------ END ------")