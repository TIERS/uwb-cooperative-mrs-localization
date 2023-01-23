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