import os
import time
import tqdm
import signal
import subprocess as sp
import dbus
sys_bus = dbus.SystemBus()
num_rounds = 5


fusion = ["only uwb", "uwb or vision", "uwb and vision"]
print("------ START ------")

for i in [0,2]:
    # print("------ {} ------".format(fusion[i]))
    for r in tqdm.tqdm(range(num_rounds)):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221011/cali_4robots_data_02"], stdout=sp.PIPE, stderr=sp.STDOUT)
        time.sleep(0.1)
        p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots_v1.3.5.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        p2 = sp.Popen(["python", "triangulation_ros2_uwb_position_v1.1.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        
        p0.wait()
        p1.send_signal(signal.SIGINT)
        p1.send_signal(signal.SIGINT)
        p2.send_signal(signal.SIGINT)
        time.sleep(1)

# ck_srv = sys_bus.get_object('org.freedesktop.ConsoleKit',
#                                 '/org/freedesktop/ConsoleKit/Manager')
# ck_iface = dbus.Interface(ck_srv, 'org.freedesktop.ConsoleKit.Manager')
# stop_method = ck_iface.get_dbus_method("Stop")
# stop_method()

# os.system("shutdown /s /t 1")
print("------ END ------")