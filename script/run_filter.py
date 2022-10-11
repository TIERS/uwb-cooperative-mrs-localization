import tqdm
import signal
import subprocess as sp

num_rounds = 2


fusion = ["only uwb", "uwb or vision", "uwb and vision"]
print("------ START ------")

tri_flag = True
for i in range(3):
    print("------ {} ------".format(fusion[i]))
    for r in tqdm.tqdm(range(num_rounds)):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221011/cali_4robots_data_02"], stdout=sp.PIPE, stderr=sp.STDOUT)
        p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        if tri_flag:
            p2 = sp.Popen(["python", "triangulation_ros2_uwb_position.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        # print("pid of ros2 bag play: {}, and pf:{}".format(p0.pid, p1.pid))
        # print("running ros2 bag")

        p0.wait()

        p1.send_signal(signal.SIGINT)
        if tri_flag:
            p2.send_signal(signal.SIGINT)
        # print("Round {} ends".format(r))
    tri_flag = False
print("------ END ------")
