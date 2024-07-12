import rclpy
from rclpy.node import Node
from sensor_msgs.msg import *
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Int32MultiArray
import cv2
from cv_bridge import CvBridge
import pickle
import os


class ColormapNode(Node):
    def __init__(self):
        super().__init__('colormap_node')

        self.lidarscan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.reference_rawdata = None
        self.rawdata = None
        self.sliced_rawdata = None
        self.normalized_data = None
        self.normalize_weight = 10
        self.scan_rawdata = None
        self.len_rawdata = 1080         # points of lidar scan
        self.angle_range = 360          # range of lidar scan
        self.angles_rawdata = np.arange(0, self.angle_range, self.angle_range/self.len_rawdata)

        self.roi_angle = [320, 40]          # -40, 40 deg
        self.roi_size = int((self.roi_angle[1] - (self.roi_angle[0]-self.angle_range))/(self.angle_range/self.len_rawdata))

        self.colormap = np.zeros([self.roi_size ,self.roi_size ])

        self.threshold_err = 0.5
        self.colormap_publisher = self.create_publisher(Image, '/colormap', 10)
        self.br = CvBridge()

        self.duration = 0.1
        self.timer = self.create_timer(self.duration, self.timer_callback)


    def timer_callback(self):
        # print(self.colormap[1:].shape)
        if self.normalized_data is not None:
            self.colormap = np.vstack((self.normalized_data[1].reshape(1,self.roi_size), self.colormap[0:-1]))
            norm = plt.Normalize(-5, 5)
            cv2_colormap = plt.cm.inferno(norm(self.colormap))
            colormap_frame = cv2.cvtColor((cv2_colormap[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            self.colormap_publisher.publish(self.br.cv2_to_imgmsg(colormap_frame, "bgr8"))
            # cv2.imshow('vis', colormap_frame)
            # cv2.waitKey(int(self.duration*1000))
            # cv2.destroyAllWindows()


            # plt.imshow(self.colormap , cmap='magma', vmin = -5, vmax = 5)
            # plt.plot(self.normalized_data[0], self.normalized_data[1])
            # plt.ylim(-3,3)
            # plt.show()
            # plt.close()

        
        


    def scan_callback(self, msg):
        self.scan_rawdata = np.array(msg.ranges)
        self.len_rawdata = len(self.scan_rawdata)
        self.rawdata = np.vstack((self.angles_rawdata, self.scan_rawdata))
        self.rawdata[1, (self.rawdata[1,:] == np.inf)] = 0
        self.sliced_rawdata = self.rawdata[:, (self.rawdata[0,:] > self.roi_angle[0])|(self.rawdata[0,:] <= self.roi_angle[1])]

        # middle_idx = int(self.sliced_rawdata.shape[1] / 2)
        # new_order = list(range(middle_idx, self.sliced_rawdata.shape[1])) + list(range(middle_idx))
        # self.sliced_rawdata = self.sliced_rawdata[:, new_order]

        if self.reference_rawdata is None:
            self.reference_rawdata = self.sliced_rawdata
        else:
            self.normalized_data = (self.sliced_rawdata/self.reference_rawdata - 1) * self.normalize_weight
            # print(self.sliced_rawdata[0])
            self.normalized_data[0] = self.sliced_rawdata[0]
            self.normalized_data = np.hstack((self.normalized_data[:, int(self.normalized_data.shape[1]/2):], self.normalized_data[:, :int(self.normalized_data.shape[1]/2)]))
            self.normalized_data[0, (self.normalized_data[0, :] > 180)] -= 360
            self.normalized_data[1, (abs(self.normalized_data[1,:]) < self.threshold_err)] = 0
            self.normalized_data[1] *= self.normalize_weight

            # print(np.mean(self.normalized_data[1]))
            # print(self.normalized_data[0:10])
            # print(self.normalized_data.shape)

        



def main(args=None):
    rclpy.init(args=args)
    cm_node = ColormapNode()
    try:
        rclpy.spin(cm_node)
    except KeyboardInterrupt:
        pass
    finally:
        cm_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()