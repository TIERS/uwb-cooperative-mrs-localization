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

tri_mems = []
pf_mems = []

def tri_mem_record(pid):
    global tri_mems
    tri_mems.clear()
    print(f"tri: {pid}")
    while True:
        tri_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
        # print(f"{len(tri_mems)}")
        time.sleep(0.1)
    
def pf_mem_record(pid):
    global pf_mems
    pf_mems.clear()    
    print(f"pf: {pid}")
    while True:
        pf_mems.append(psutil.Process(pid).memory_info().rss / 1024 ** 2)
        time.sleep(0.1)


bag_files = ["20221011/cali_4robots_data_02"]
#  "20221013/cali_4robots_data_04", "20221013/4robots_data_07"]
for bag_file in bag_files:
    print(f"------Bag File: {bag_file}------")
    tri_flag = True
    for i, fus in enumerate(fusion): 
        print(f"------Fusion Type: {fus}------")
        for r in tqdm.tqdm(range(num_rounds)):
            print("//////////////////////")
            p0 = sp.Popen(["ros2", "bag", "play", "recorded_data/{}".format(bag_file)], stdout=sp.PIPE, stderr=sp.STDOUT)
            time.sleep(0.1)
            p1 = sp.Popen(["python", "pfilter_ros2_uwb_position_single_range.py", "--fuse_group", "{}".format(i), "--with_polyfit", "{}".format(True), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

            p2 = sp.Popen(["python", "pfilter_ros2_uwb_position_single_range.py", "--fuse_group", "{}".format(i), "--with_polyfit", "{}".format(False), "--round", "{}".format(r)], stdout=sp.PIPE, stderr=sp.STDOUT)

            if os.path.exists("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)):
                shutil.rmtree("./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)) 
            # time.sleep(100)
            # time.sleep(150)
            p3 = sp.Popen(["ros2", "bag", "record",  "/real_turtle04_pose",  "/pf_turtle04_pose", "/pf_poly_turtle04_pose", "-o", "./result_bags/{}/result_bag_{}_{}".format(bag_file, i, r)], stdout=sp.PIPE, stderr=sp.STDOUT)
            # p4 = sp.Popen([""])
            # print(f"/////// {p2.pid}")

            # tri = mlp.Process(target=tri_mem_record, args=(p2.pid,))
            # pf = mlp.Process(target=pf_mem_record, args=(p1.pid,))
            
            # tri.start()
            # pf.start()
            p0.wait()
            # global tri_mems
            # global pf_mems
            # tri.terminate()
            # pf.terminate()
            
            # print(f"////// tri_mems: {tri_mems}")
            # print(f"////// pf_mems: {pf_mems}")

            # np.savetxt("./results/triangulation/computation/mem_tri_{}.csv".format(r), np.array(tri_mems))
            # np.savetxt("./results/pfilter/computation/mem_pf_{}_{}.csv".format(fus, r), np.array(pf_mems))
            p1.send_signal(signal.SIGINT)
            p2.send_signal(signal.SIGINT)
            p3.send_signal(signal.SIGINT)



            time.sleep(1)

print("------ END ------")