import os
import subprocess as sp

num_rounds = 1


fusion = ["only uwb", "uwb or vision", "uwb and vision"]
for i in range(3):
    print("------ {} ------".format(fusion[i]))
    for r in range(num_rounds):
        p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221004/4robots_data_01"], stdout=sp.PIPE)
        p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)
        print(p0.pid)
        print("running ros2 bag")
        p0.communicate() # wait until session end
        p0.kill()
        p1.kill()
        # p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
        # os.system("python pfilter_ros2_multi_robots_only_uwb.py --fuse_group {} --round {}".format(i,r))
        
        
        # p0.wait()
        # p2 = sp.run(["kill", "-9", "{}".format(p1.pid)])
        # p2 = sp.run(["ros2", "lifecycle", "set", "relative_pf_rclpy", "shutdown"])
        # p1.terminate()
        # print("round {} ends".format(r))
        # while p0.poll() is None:
        #     p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
        #     p0.wait()
        
        # p2 = sp.run(["ros2", "lifecycle", "set", "relative_pf_rclpy", "shutdown"])
        # p1.kill()

        print("round {} ends".format(r))
        #     print("end ----------------")




# fusion = ["only uwb", "uwb or vision", "uwb and vision"]
# for i in range(3):
#     print("------ {} ------".format(fusion[i]))
#     for r in range(num_rounds):
#         p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/20221004/4robots_data_01"], stdout=sp.PIPE)
#         p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
#         print(p0.pid)

#         print(p1.pid)
#         # p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
#         # os.system("python pfilter_ros2_multi_robots_only_uwb.py --fuse_group {} --round {}".format(i,r))
#         print("running ros2 bag")
#         p0.wait()
#         p2 = sp.run(["kill", "-9", "{}".format(p1.pid)])
#         # p2 = sp.run(["ros2", "lifecycle", "set", "relative_pf_rclpy", "shutdown"])
#         # p1.terminate()
#         # print("round {} ends".format(r))
#         # while p0.poll() is None:
#         #     p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE)
#         #     p0.wait()
        
#         # p2 = sp.run(["ros2", "lifecycle", "set", "relative_pf_rclpy", "shutdown"])
#         # p1.kill()

#         print("round {} ends".format(r))
#         #     print("end ----------------")