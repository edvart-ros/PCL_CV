#!/usr/bin/python3

import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, PointCloud2
import rospy
import time
import numpy as np
import time
from tf import TransformListener
from ultralytics import YOLO
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from utils import *
import ros_numpy
import cv2

MODEL_PATH = '/home/navier_orin/orin_ws/src/navier_orin/navier_perception/navier_object_detection/weights/segment_0_1.pt'
CAMERA_FRAME = 'zedx_left_camera_optical_frame'
ODOM_FRAME = 'odom'


class ZedDetector():
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH, task='detect')
        self.color_map_rgba_norm = {0: (0, 1, 0, 1), 1: (1, 0, 0, 1), 2: (1, 1, 0, 1)} # green, red, yellow
        self.color_map_rgb = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 0, 255)}

        
        self.cam_info_sub = rospy.Subscriber('/zedx/zed_node/rgb/camera_info', CameraInfo, self.cam_info_callback)
        self.depth_sub = rospy.Subscriber('/zedx/zed_node/depth/depth_registered', Image, self.depth_callback, queue_size=1)
        cam_info = rospy.wait_for_message('/zedx/zed_node/rgb/camera_info', CameraInfo)
        self.fx = cam_info.K[0]
        self.cx = cam_info.K[2]
        self.fy = cam_info.K[4]
        self.cy = cam_info.K[5]

        self.marker_pub = rospy.Publisher('/visualization_marker', MarkerArray, queue_size=10)
        self.velo_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.velo_cb)
        self.segment_result_pub = rospy.Publisher('/zedx/zed_node/rgb/image_segmentations/compressed', CompressedImage, queue_size=10)
        self.rgb_sub = rospy.Subscriber('/zedx/zed_node/rgb/image_rect_color', Image, self.rgb_callback, queue_size=1)


    def rgb_callback(self, rgb_msg):
        fusion_pixels, fusion_points = np.copy(self.fusion_pixels), np.copy(self.fusion_points)
        depth = np.copy(self.depth)
        img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        result = self.model(img, verbose=True)[0]

        if result.masks is None:
            return
        
        masks_classes = result.boxes.cls.cpu().numpy()
        masks_corners = result.masks.xy
        
        # estimate object locations
        object_positions = [] 
        object_classes = [] 
        for corners, detection_class in zip(masks_corners, masks_classes):
            img = cv2.polylines(img, [corners.astype()], isClosed=True, color=self.color_map_rgb.get(detection_class), thickness=3)
            
            #retrieves a boolean mask of the object in the image
            poly_mask = self.get_mask_from_poly_corners(corners, depth)
            # estimate the position of the object using the object mask and depth image
            stereo_pos_estimate = self.stereo_estimate_object_position(poly_mask, depth)
            # estimate the position of the object using the mask and lidar pointcloud
            # ???
            #
            if not np.all(np.isnan(stereo_pos_estimate)):
                object_positions.append(stereo_pos_estimate)
                object_classes.append(detection_class)
        
        self.compress_and_publish_img(img, self.segment_result_pub, 'jpg', CAMERA_FRAME)

        # visualize object positions with rviz markers
        marker_array = self.get_points_marker_array(object_positions, masks_classes)
        self.marker_pub.publish(marker_array)


    def depth_callback(self, depth_msg):
        self.depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')


    def velo_callback(self, pcl2_msg):
        self.fusion_pixels, self.fusion_points = self.get_projections(pcl2_msg)


    def get_projections(self, pcl2_msg):
        pcl = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl2_msg)
        pcl = pcl.transpose() #transpose to get one point per column

        points = np.dot(self.R, pcl) + self.translation  #rotate into camera frame then translate
        points = points[:, np.where(points[2, :] >= 0)[0]] #pick out points in front of the camera
        homogeneous_points = np.vstack((points, np.ones((1, points.shape[1])))) #add homogeneous coordinates
        homogeneous_projections = np.dot(self.P, homogeneous_points) #project onto camera
        pixels = homogeneous_projections[:2] / homogeneous_projections[2:] #normalize and remove homogeneous coordinate

        #remove projections that land outside of the camera resolution
        in_bounds = (pixels[0, :] > 0) & (pixels[0, :] < self.cam_info.width-1) & (pixels[1, :] > 0) & (pixels[1, :] < self.cam_info.height-1) #get indexes within bounds of camera resolution
        pixels = (np.rint(pixels[:, in_bounds])).astype(int) #pixels that are in our cameras image
        points = points[:,  in_bounds] # corresponding points in 3d
        image_points = np.zeros((np.shape(self.img)))
        image_points[pixels] = points

        #stack such that each array element represents one point, and one pixel
        pixels = np.column_stack(pixels)
        points = np.column_stack(points)
        return pixels, points


    def get_points_marker_array(self, positions, object_classes):
        markers = []
        for i in range(len(positions)):
            marker = Marker()
            marker.header.frame_id = CAMERA_FRAME
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(0.1)
            marker.id = i

            marker.type = 2
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15

            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.color_map_rgba_norm.get(object_classes[i], (0, 0, 0, 1))

            marker.pose.orientation.w = 1.0
            marker.pose.position.x = positions[i][0]
            marker.pose.position.y = positions[i][1]
            marker.pose.position.z = positions[i][2]
            markers.append(marker)

        marker_array_msg = MarkerArray()
        marker_array_msg.markers = markers
        return marker_array_msg

    def stereo_estimate_object_position(self, poly_mask, depth):
        depth_values = depth[poly_mask]
        y_indices, x_indices = np.indices(poly_mask.shape)
        y_indices = y_indices[poly_mask]
        x_indices = x_indices[poly_mask]

        points = np.array([(x_indices - self.cx) * depth_values / self.fx, \
                           (y_indices - self.cy) * depth_values / self.fy, \
                            depth_values])
        pos_estimate = np.nanmedian(points, axis=1)
        return pos_estimate
    

    def get_mask_from_poly_corners(self, corners, depth):
        poly_mask = np.zeros_like(depth)
        mask = mask.astype(int)
        cv2.fillPoly(poly_mask, [mask], 1)
        poly_mask = poly_mask.astype(bool)
        return poly_mask


    def estimate_object_position(self, mask, depth):
        poly_mask = np.zeros_like(depth)  # single channel mask
        mask = mask.astype(int)
        cv2.fillPoly(poly_mask, [mask], 1)
        poly_mask = poly_mask.astype(bool)

        depth_values = depth[poly_mask]
        y_indices, x_indices = np.indices(poly_mask.shape)
        y_indices = y_indices[poly_mask]
        x_indices = x_indices[poly_mask]

        points = np.array([(x_indices - self.cx) * depth_values / self.fx, \
                           (y_indices - self.cy) * depth_values / self.fy, \
                            depth_values])
        pos_estimate = np.nanmedian(points, axis=1)
        return pos_estimate


    def compress_and_publish_img(self, image, publisher, format='jpg', frame=CAMERA_FRAME):
        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image, format)
        compressed_msg.header.frame_id = frame
        publisher.publish(compressed_msg)


    def cam_info_callback(self, info_msg):
        self.fx = info_msg.K[0]
        self.cx = info_msg.K[2]
        self.fy = info_msg.K[4]
        self.cy = info_msg.K[5]





if __name__ == "__main__":
    rospy.init_node("zed_detector")
    node = ZedDetector()
    rospy.spin()