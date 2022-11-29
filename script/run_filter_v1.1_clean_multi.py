import os
import time
import tqdm
import signal
import shutil
import subprocess as sp
import threading
import numpy as np
num_rounds = 1
import os, psutil
import multiprocessing as mlp

fusion = ["uwb", "uwb_vision"]
print("------ START ------")


# bag_file = "cali_4robots_data_02"
# bag_file = "cali_4robots_data_04"
# bag_file = "4robots_data_07"

# global tri_mems

with mlp.Manager() as manager:

    def tri_mem_record(pid, tri_mems):
        while True:
            tri_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
            time.sleep(0.1)
        
    def pf_mem_record(pid, pf_mems):
        while True:
            pf_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
            time.sleep(0.1)


    # bag_file = "cali_4robots_data_02"
    # bag_file = "cali_4robots_data_04"
    # bag_file = "4robots_data_07"
    bag_files = ["20221011/cali_4robots_data_02"]
    # bag_files = ["20221011/shortbag"]
    #  "20221013/cali_4robots_data_04", "20221013/4robots_data_07"]
    # bag_files = ["20221013/4robots_data_07"] 
    for bag_file in bag_files:
        print(f"------Bag File: {bag_file}------")
        tri_flag = True
        for i, fus in enumerate(fusion): 
            tri_mems = manager.list([])
            pf_mems = manager.list([])
            print(f"------Fusion Type: {fus}------")
            for r in tqdm.tqdm(range(num_rounds)):
                print("//////////////////////")
                p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/{}".format(bag_file)], stdout=sp.PIPE, stderr=sp.STDOUT)
                time.sleep(0.1)
                p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots_v5.0.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

                p2 = sp.Popen(["python", "triangulation_ros2_uwb_position_v3.0.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

                if os.path.exists("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)):
                    shutil.rmtree("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)) 
                time.sleep(100)
                # time.sleep(150)
                p3 = sp.Popen(["ros2", "bag", "record",  "/tri_turtle01_pose", "/tri_turtle03_pose", "/tri_turtle04_pose", "/real_turtle01_pose", "/real_turtle03_pose",
                        "/real_turtle04_pose", "/real_turtle05_pose", "/pf_turtle01_pose","/pf_turtle03_pose", "/pf_turtle04_pose", "-o", "./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)], stdout=sp.PIPE, stderr=sp.STDOUT)
                # p4 = sp.Popen([""])
                # print(f"/////// {p2.pid}")
                # tri_que = mlp.Queue()
                # pf_que  = mlp.Queue()
                tri = mlp.Process(target=tri_mem_record, args=(p2.pid,tri_mems,))
                pf = mlp.Process(target=pf_mem_record, args=(p1.pid,pf_mems,))
                
                tri.start()
                pf.start()
                p0.wait()
                # global tri_mems
                # global pf_mems
                # print(f"---- {pf_que.get()}")
                tri.terminate()
                pf.terminate()
                
                print(f"////// tri_mems: {tri_mems}")
                print(f"////// pf_mems: {pf_mems}")

                np.savetxt("./results/triangulation/computation/mem_tri_{}.csv".format(r), np.array(tri_mems))
                np.savetxt("./results/pfilter/computation/mem_pf_{}_{}.csv".format(fus, r), np.array(pf_mems))
                p1.send_signal(signal.SIGINT)
                p2.send_signal(signal.SIGINT)
                p3.send_signal(signal.SIGINT)

                time.sleep(1)

    print("------ END ------")