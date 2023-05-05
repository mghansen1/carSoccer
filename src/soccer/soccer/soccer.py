import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

from enum import Enum

TIMER_INTERVAL = 0.1


class MoveState(Enum):
    FORWARD = 0


class Soccer(Node):
    def __init__(self):
        super().__init__('soccer')
        self.bridge = CvBridge()

        self.move_publisher = self.create_publisher(Twist, 'zelda/cmd_vel', 10)
        self.move_timer = self.create_timer(
            TIMER_INTERVAL,
            self.move_timer_callback
        )

        self.move_state = MoveState.FORWARD

        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos.qos_profile_sensor_data
        )
        self.camera_subscription

    def move_timer_callback(self):
        pass

    def image_callback(self, msg: Image):
        self.get_logger().debug("got frame")

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        cv2.imshow('image', img)

        balls = self.detect_balls(img)

        cv2.waitKey(1)

    def detect_balls(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellowLower = (25, 40, 100)
        yellowUpper = (80, 255, 255)

        mask = cv2.inRange(hsv, yellowLower, yellowUpper)

        # cv2.imshow('mask', mask)

        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        return output


def main(args=None):
    rclpy.init(args=args)

    soccer = Soccer()

    rclpy.spin(soccer)

    soccer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
