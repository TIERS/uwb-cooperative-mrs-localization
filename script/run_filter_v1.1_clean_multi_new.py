import os
import time
import tqdm
import signal
import shutil
import subprocess as sp
import numpy as np
import os, psutil
import multiprocessing as mlp

num_rounds = 1
fusion = ["uwb", "uwb_vision"]
print("------ START ------")

with mlp.Manager() as manager:

    def tri_mem_record(pid, tri_mems):
        while True:
            tri_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
            time.sleep(0.1)
        
    def pf_mem_record(pid, pf_mems):
        while True:
            pf_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
            time.sleep(0.1)

    bag_files = ["20221217/rosbag2_20_06_59"]

    # bag_file = "4robots_data_07"
    # bag_files = ["20221011/cali_4robots_data_02"]
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
                p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_multi_robots_v5.0_new_fake_odom.py", "--fuse_group", "{}".format(i), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

                p2 = sp.Popen(["python", "triangulation_ros2_uwb_position_v3.0_new.py", "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

                if os.path.exists("./results/result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)):
                    shutil.rmtree("./results/result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)) 
                time.sleep(30)  
                # time.sleep(150)
                p3 = sp.Popen(["ros2", "bag", "record",  "/tri_turtle01_pose", "/tri_turtle02_pose", "/tri_turtle03_pose", "/tri_turtle05_pose", "/real_turtle01_pose", "/real_turtle02_pose", "/real_turtle03_pose",
                        "/real_turtle04_pose", "/real_turtle05_pose", "/pf_turtle01_pose","/pf_turtle02_pose","/pf_turtle03_pose", "/pf_turtle05_pose", "-o", "./results/result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)], stdout=sp.PIPE, stderr=sp.STDOUT)
                # p4 = sp.Popen([""])
                # print(f"/////// {p2.pid}")
                # tri_que = mlp.Queue()
                # pf_que  = mlp.Queue()
                tri = mlp.Process(target=tri_mem_record, args=(p2.pid,tri_mems,))
                pf = mlp.Process(target=pf_mem_record, args=(p1.pid,pf_mems,))
                
                tri.start()
                pf.start()
                p0.wait()
                # print(f"---- {pf_que.get()}")
                tri.terminate()
                pf.terminate()
                
                np.savetxt("./results/results_csv/triangulation/computation/mem_tri_{}.csv".format(r), np.array(tri_mems))
                np.savetxt("./results/results_csv/pfilter/computation/mem_pf_{}_{}.csv".format(fus, r), np.array(pf_mems))
                p1.send_signal(signal.SIGINT)
                p2.send_signal(signal.SIGINT)
                p3.send_signal(signal.SIGINT)

                time.sleep(1)

    print("------ END ------")