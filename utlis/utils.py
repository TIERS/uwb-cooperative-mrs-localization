import argparse
import math
from scipy.spatial.transform    import Rotation as R


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def euler_from_quaternion(pose):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = pose[0], pose[1], pose[2],pose[3]
    # x =  pose.x
    # y =  pose.y
    # z =  pose.z
    # w =  pose.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    # return roll_x, pitch_y, yaw_z # in radians
    return yaw_z


def cal_yaws(odom):
    r = R.from_quat([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
    yaw, _, _ = r.as_euler('zxy', degrees=True)
    return yaw

def cal_yaws(array):
    r = R.from_quat(array)
    yaw, _, _ = r.as_euler('zxy', degrees=False)
    return yaw


def dict_values_empty(dict):
    return all(ele.size > 0 for ele in dict.values())


def dicts_empty(dicts):
    return all(dict_values_empty(dict) for dict in dicts)


odom_transform = {
    0: ([[ 1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]],[0.0,0.0,0.0]),
    1: ([[ 1.00000000e+00, -7.58541994e-16,  5.13445841e-16],
         [ 5.62165047e-16,  1.00000000e+00,  6.07270648e-17],
         [-5.11982838e-16, -5.43282007e-17,  1.00000000e+00]], [[ 8.88178420e-14,  6.19504448e-14, -2.74780199e-15]]),
    2: ([[ 1.00000000e+00, -1.19249603e-06,  5.97508164e-07],
         [ 1.19249650e-06,  1.00000000e+00, -7.73259505e-07],
         [-5.97507242e-07,  7.73260217e-07,  1.00000000e+00]], [[ 7.78583451e-06, -9.16588829e-06, -3.91865132e-07]]),
    3: ([[ 1.00000000e+00,  6.95738684e-16,  5.84436962e-17],
         [-7.08797061e-16,  1.00000000e+00,  5.65593577e-16],
         [-5.79974105e-17, -5.61957041e-16,  1.00000000e+00]], [[-3.24185123e-14,  7.99360578e-14, -1.72084569e-15]]),
    4: ([[ 1.00000000e+00,  2.86754311e-05,  7.06831308e-07],
         [-2.86754311e-05,  1.00000000e+00, -1.37164463e-07],
         [-7.06835241e-07,  1.37144194e-07,  1.00000000e+00]], [[ 1.75782436e-04, -1.46837665e-04, -1.94010446e-06]])
}