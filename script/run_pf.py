import os
import subprocess as sp

num_rounds = 1


fusion = ["only uwb", "uwb or vision", "uwb and vision"]
for i in range(3):
    print("------ {} ------".format(fusion[i]))
    for r in range(num_rounds):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221004/4robots_data_01"], stdout=sp.PIPE)
        # p1 = sp.Popen(["python", "pfilter_ros2_multi_robots_only_uwb.py", "--fuse_group {}".format(i), "--round {}".format(r)], stdout=sp.PIPE)
        p1 = sp.Popen(["python", "pfilter_ros2_multi_robots_only_uwb.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
        # os.system("python pfilter_ros2_multi_robots_only_uwb.py --fuse_group {} --round {}".format(i,r))

        p0.wait()
        p1.terminate()
        print("round {} ends".format(r))
        # while p0.poll() is None:
        #     print("end ----------------")