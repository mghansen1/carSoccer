import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from apriltag_msgs.msg import AprilTagDetectionArray

import math
from collections import deque

import cv2
import numpy as np
from cv_bridge import CvBridge

from enum import Enum

from super_gradients.training import models
from super_gradients.common.object_names import Models

from dataclasses import dataclass
from typing import Tuple, Optional

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# DO_MOVE = False
DO_MOVE = True
TIMER_INTERVAL = 0.1
TF_TIMER_INTERVAL = 0.5
MOVE_SPEED = 0.25
TURN_POWER = 1.5
IMAGE_DIMS = np.array([320, 240])

APRILTAG_POSITIONS = {
    "tag16h5:0": (-0.61, 2),
    "tag16h5:1": (0, 2.33),
    "tag16h5:2": (0.61, 2),
    "tag16h5:3": (1.5, 0),
    "tag16h5:4": (0.61, -2),
    "tag16h5:5": (0, -2.33),
    "tag16h5:6": (-0.61, -2),
    "tag16h5:7": (-1.5, 0),
}


def y_to_dist(y: float):
    a = 157.602
    b = -0.317125
    c = -155.118
    dist = a * (y**b) + c
    return dist / 100.0


def prop(error, k, max_power, min_error: float = 0):
    out = np.clip(error * k, -max_power, max_power)

    if abs(error) < min_error:
        out = 0.0

    return float(out)


def to_np(p: Point):
    return np.array([p.x, p.y, p.z])


# Taken from: https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434
def euler_from_quaternion(quaternion: Quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # make sure the angles are in [0, 2*pi]
    yaw = (yaw + 4 * np.pi) % (2 * np.pi)

    return np.array([roll, pitch, yaw])

def rotate_about_origin(point: np.ndarray, angle: float) -> np.ndarray:
    rot = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])

    rotated = rot @ point[:2]

    return np.array([rotated[0], rotated[1], point[2]])


class MoveState(Enum):
    FORWARD = 0
    STOPPED = 1
    ALIGN_TO_BALL = 2
    NAV_TO_POINT = 3
    ALIGN_TO_APRILTAG = 4

class NavPlanStep(Enum):
    TURN = 0
    FORWARD = 1

@dataclass
class Pose:
    position: np.ndarray
    rotation: float

@dataclass
class Prediction:
    bbox: Tuple[int]
    centroid: Tuple[int]
    label: str

ROBOT_NAME = "yoshi"

class Soccer(Node):
    def __init__(self):

        super().__init__("zelda_py_soccer")
        self.bridge = CvBridge()

        self.move_publisher = self.create_publisher(Twist, f"{ROBOT_NAME}/cmd_vel", 10)
        self.move_timer = self.create_timer(TIMER_INTERVAL, self.move_timer_callback)

        self.yolo = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
        self.label_set = set()

        self.move_state = MoveState.NAV_TO_POINT

        self.forward = 0.0
        self.turn = 0.0

        self.plan: deque[Pose] = deque([
            Pose(np.array([-1.0, -1.0, 0.0]), math.radians(0.0)),
            Pose(np.array([0.5, -0.7, 0.0]), math.radians(30.0)),
        ])

        self.camera_subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos.qos_profile_sensor_data,
        )
        self.camera_subscription

        self.odom_subscription = self.create_subscription(
            Odometry, f"/{ROBOT_NAME}/odom", self.odom_callback, qos.qos_profile_sensor_data
        )
        self.odom_subscription

        self.base_position = None
        self.base_orientation = None
        self.position = None
        self.orientation = None

        self.nav_start_position = None
        self.nav_start_orientation = None

        self.apriltag_subscription = self.create_subscription(
            AprilTagDetectionArray,
            "/detections",
            self.apriltag_callback,
            qos.qos_profile_sensor_data,
        )

        self.detections = {}

        self.nav_plan = None

    def apriltag_callback(self, msg):
        self.detections = {}

        for d in msg.detections:
            center = np.array([d.centre.x, d.centre.y]) / IMAGE_DIMS
            self.detections[d.id] = center

        if self.move_state == MoveState.ALIGN_TO_APRILTAG:
            if 1 in self.detections:
                print("ALIGNING TO APRIL TAG")
                pos = self.detections[1]

                x_error = 0.5 - pos[0]
                self.turn = prop(x_error, 0.8, 0.3)

                if abs(x_error) < 0.05:
                    self.move_state = MoveState.STOPPED
            else:
                self.turn = 0.05

    def move_timer_callback(self):
        msg = Twist()


        match self.move_state:
            case MoveState.FORWARD:
                pass
            case MoveState.STOPPED:
                pass
            case MoveState.ALIGN_TO_BALL | MoveState.ALIGN_TO_APRILTAG:
                msg.angular.z = self.turn
            case MoveState.NAV_TO_POINT:
                msg.linear.x = self.forward
                msg.angular.z = self.turn

        # print(msg)

        if DO_MOVE:
            self.move_publisher.publish(msg)

    def turn_timer_callback(self):
        self.move_state = MoveState.STOPPED

    def odom_callback(self, odom: Odometry):
        raw_position = to_np(odom.pose.pose.position)
        raw_orientation = euler_from_quaternion(odom.pose.pose.orientation)

        if self.base_position is None:
            self.base_position = raw_position

        if self.base_orientation is None:
            self.base_orientation = raw_orientation

        self.position = rotate_about_origin(raw_position - self.base_position, -self.base_orientation[2]) + np.array([-2.0, 0.0, 0.0])
        self.orientation = raw_orientation - self.base_orientation


        if self.move_state == MoveState.NAV_TO_POINT:
            goal = self.plan[0]

            if self.nav_plan is None:
                # create a nav plan
                # first, rotate towards target position
                first_angle = math.atan2(goal.position[1] - self.position[1], goal.position[0] - self.position[0])

                # next, move forward until we reach the target position
                distance = np.linalg.norm(goal.position[:2] - self.position[:2])

                # finally, rotate towards the target orientation
                second_angle = goal.rotation

                self.nav_plan = deque([
                    (NavPlanStep.TURN, first_angle),
                    (NavPlanStep.FORWARD, distance),
                    (NavPlanStep.TURN, second_angle)
                ])

                self.nav_start_position = self.position
                self.nav_start_orientation = self.orientation

            if len(self.nav_plan) == 0:
                self.nav_plan = None
                self.plan.popleft()
                return
            
            step = self.nav_plan[0]
            kind, value = step
            done = False

            print(f"nav plan step: {kind}, {value}")

            if kind == NavPlanStep.TURN:
                self.forward = 0.0
                self.turn = -prop(self.orientation[2] - value, 1.5, 1.0)

                if abs(self.orientation[2] - value) < math.radians(5):
                    done = True

            elif kind == NavPlanStep.FORWARD:
                dist = np.linalg.norm(self.position[:2] - self.nav_start_position[:2])
                self.forward = prop(value - dist, 0.9, 0.13)
                self.turn = 0.0

                if dist > value:
                    done = True

            if done:
                self.nav_plan.popleft()

        print(self.position, self.orientation)

    def image_callback(self, msg: Image):
        self.get_logger().debug("got frame")

        # PERCEPTION

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        yellowRange = ((25, 40, 100), (80, 255, 255))
        yellow_balls = self.detect_balls(img, yellowRange, 20)
        blue_ball = self.find_blue_ball(img)

        # draw a dot on each ball, decreasing in color intensity
        # for i, ball in enumerate(yellow_balls):
        #     cv2.circle(img, (int(ball[0]), int(ball[1])), 10, (0, 255, 0), -1)

        # if blue_ball is not None:
        #     x, y = blue_ball.centroid
        #     cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), -1)

        print(self.detections)
        for detection in self.detections.values():
            x, y = detection * IMAGE_DIMS
            cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), -1)

        self.show_img(img, "Camera")

        cv2.waitKey(1)

        # CONTROL/PLANNING

        if self.position is None or self.orientation is None:
            return

        if (
            self.move_state == MoveState.FORWARD
            or self.move_state == MoveState.ALIGN_TO_BALL
        ):
            if len(yellow_balls) > 0:
                yellow_balls = yellow_balls / IMAGE_DIMS  # normalize

                target = yellow_balls[0]
                x_error = 0.5 - target[0]
                self.turn = prop(x_error, 0.8, 0.3)

                if abs(x_error) < 0.05:
                    self.move_state = MoveState.ALIGN_TO_APRILTAG
                    self.ball_angle = self.orientation[2]
            else:
                self.turn = 0.0

    def detect_balls(self, img, colorRange, area_min, name="mask") -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower, upper = colorRange
        mask = cv2.inRange(hsv, lower, upper)

        # erode and dilate to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        self.show_img(mask, name)

        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        idx = stats[:, cv2.CC_STAT_AREA] > area_min
        idx[0] = False  # ignore the background
        stats = stats[idx]
        centroids = centroids[idx]

        # Sort by area
        idx = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
        stats = stats[idx]
        centroids = centroids[idx]

        return centroids

    def detect_objects(self, img) -> list[Prediction]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = self.yolo.predict(img)

        out_preds = []

        for pred in preds:
            class_names = pred.class_names
            pred = pred.prediction
            labels = pred.labels

            for pred_i in range(len(pred.labels)):
                x1 = int(pred.bboxes_xyxy[pred_i, 0])
                y1 = int(pred.bboxes_xyxy[pred_i, 1])
                x2 = int(pred.bboxes_xyxy[pred_i, 2])
                y2 = int(pred.bboxes_xyxy[pred_i, 3])

                label = class_names[int(labels[pred_i])]
                self.label_set.add(label)
                centroid = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                )

                out_preds.append(
                    Prediction(bbox=(x1, y1, x2, y2), label=label, centroid=centroid)
                )

                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.show_img(img, "Objects")

        return out_preds

    def find_blue_ball(self, img) -> Optional[Prediction]:
        preds = self.detect_objects(img)

        for p in preds:
            if p.label == "sports ball":
                x1, y1, x2, y2 = p.bbox
                bbox_area = img[y1:y2, x1:x2]
                avg_color = np.mean(bbox_area, axis=(0, 1))[::-1]

                if self.is_blue(avg_color):
                    return p

        return None

    def is_blue(self, rgb):
        # Convert RGB to HSV color space
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
        # Define range of dark blue color in HSV
        lower_blue = (90, 30, 20)
        upper_blue = (130, 255, 200)
        # Check if the HSV value is within the range of dark blue color
        return cv2.inRange(hsv, lower_blue, upper_blue)[0][0] == 255

    def show_img(self, img, title=None, text=None):
        window_name = title if title else "image"
        new_img = img.copy()

        # add title text
        if title:
            cv2.putText(
                new_img,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        if text:
            cv2.putText(
                new_img,
                text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow(window_name, new_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    soccer = Soccer()

    rclpy.spin(soccer)

    soccer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
