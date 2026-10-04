#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gripper_joint_state_publisher.py — Publish JointState cho 2 ngón kẹp vật lý (Bài 03).

Chức năng:
  - Publish liên tục (30 Hz) trạng thái khớp của left_finger_joint và right_finger_joint
    lên topic /joint_states từ giây 0.
  - Cập nhật vị trí tức thời khi nhận lệnh từ /gripper/left_cmd và /gripper/right_cmd.
  - Loại bỏ hoàn toàn cảnh báo 'Missing left_finger_joint, right_finger_joint' của MoveGroup.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class GripperJointStatePublisher(Node):
    def __init__(self):
        super().__init__("gripper_joint_state_publisher")
        
        self.left_pos = 0.018   # Khởi tạo ở trạng thái mở rộng
        self.right_pos = -0.018

        self.target_left_pos = 0.018
        self.target_right_pos = -0.018

        # Publisher /joint_states
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        # Subscribers
        self.create_subscription(Float64, "/gripper/left_cmd", self._on_left_cmd, 10)
        self.create_subscription(Float64, "/gripper/right_cmd", self._on_right_cmd, 10)

        # Timer 30 Hz (~33ms)
        self.timer = self.create_timer(0.033, self._publish_joint_state)
        self.get_logger().info("GripperJointStatePublisher đã khởi chạy (30 Hz).")

    def _on_left_cmd(self, msg: Float64):
        self.target_left_pos = float(msg.data)

    def _on_right_cmd(self, msg: Float64):
        self.target_right_pos = float(msg.data)

    def _publish_joint_state(self):
        # Nội suy mượt mà tiến về target
        self.left_pos += (self.target_left_pos - self.left_pos) * 0.35
        self.right_pos += (self.target_right_pos - self.right_pos) * 0.35

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["left_finger_joint", "right_finger_joint"]
        msg.position = [float(self.left_pos), float(self.right_pos)]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GripperJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
