import os
import tqdm
import signal
import subprocess as sp

num_rounds = 5


fusion = ["only uwb", "uwb or vision", "uwb and vision"]
print("------ START ------")

tri_flag = True
for i in [0,2]:
    # print("------ {} ------".format(fusion[i]))
    for r in tqdm.tqdm(range(num_rounds)):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221011/cali_4robots_data_02"], stdout=sp.PIPE, stderr=sp.STDOUT)
        p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots_v1.3.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        if tri_flag:
            p2 = sp.Popen(["python", "triangulation_ros2_uwb_position.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        p0.wait()

        p1.send_signal(signal.SIGINT)
        if tri_flag:
            p2.send_signal(signal.SIGINT)
    tri_flag = False


# os.system("shutdown /s /t 1")
print("------ END ------")
