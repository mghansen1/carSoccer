import rclpy
from rclpy.node import Node
import rclpy
from rclpy import qos

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge

from enum import Enum

import super_gradients

from super_gradients.training import models
from super_gradients.common.object_names import Models

from dataclasses import dataclass
from typing import Tuple, Optional


TIMER_INTERVAL = 0.1


class MoveState(Enum):
    FORWARD = 0


@dataclass
class Prediction:
    bbox: Tuple[int]
    centroid: Tuple[int]
    label: str


class Soccer(Node):
    def __init__(self):
        super().__init__('soccer')
        self.bridge = CvBridge()

        self.move_publisher = self.create_publisher(Twist, 'zelda/cmd_vel', 10)
        self.move_timer = self.create_timer(
            TIMER_INTERVAL,
            self.move_timer_callback
        )

        self.yolo = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
        self.label_set = set()

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

        yellowRange = (
            (25, 40, 100),
            (80, 255, 255)
        )
        yellow_balls = self.detect_balls(img, yellowRange, 20)
        blue_ball = self.find_blue_ball(img)

        # blueRange = (
        #     (105, 40, 20),
        #     (140, 255, 200)
        # )
        # blue_ball = self.detect_balls(img, blueRange, 30, name="blue_mask")

        # draw a dot on each ball, decreasing in color intensity
        for i, ball in enumerate(yellow_balls):
            cv2.circle(
                img,
                (int(ball[0]), int(ball[1])),
                10,
                (i * 100, 255 - i * 50, 0),
                -1
            )

        self.show_img(img, "Camera")

        cv2.waitKey(1)

    def detect_balls(self, img, colorRange, area_min, name="mask") -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower, upper = colorRange
        mask = cv2.inRange(hsv, lower, upper)

        # erode and dilate to remove noise
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        self.show_img(mask, name)

        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        idx = stats[:, cv2.CC_STAT_AREA] > area_min
        idx[0] = False  # ignore the background
        stats = stats[idx]
        centroids = centroids[idx]

        # Sort by area
        idx = np.argsort(stats[:, cv2.CC_STAT_AREA])
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
                    Prediction(bbox=(x1, y1, x2, y2),
                               label=label, centroid=centroid)
                )

                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.show_img(img, "Objects")

        print(self.label_set)

        return out_preds

    def find_blue_ball(self, img) -> Optional[Prediction]:
        preds = self.detect_objects(img)

        for p in preds:
            x1, y1, x2, y2 = p.bbox
            bbox_area = img[x1:x2, y1:y2]
            print(bbox_area.shape)

        return None

    def show_img(self, img, title=None):
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
                2
            )

        cv2.imshow(window_name, new_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    soccer = Soccer()

    rclpy.spin(soccer)

    soccer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
