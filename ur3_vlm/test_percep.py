#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
from ur3_vlm.camera_perception import CameraPerception

def main():
    rclpy.init()
    node = Node("perception_tester")

    with open("src/ur3_vlm/config/scene.yaml") as f:
        cfg = yaml.safe_load(f)

    percep = CameraPerception(cfg)
    bridge = CvBridge()

    def img_cb(msg):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        world_state, annotated = percep.process_image(cv_img)
        cv2.imwrite("/tmp/test_annotated.jpg", annotated)
        print("=== DETECTED OBJECTS ===")
        for name, data in world_state["detected_objects"].items():
            px = data["pixel"]
            pos = data["position"]
            print(f"{name:12s}: Pixel=({px[0]:.1f}, {px[1]:.1f}) -> Est Pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        print("\n=== DETECTED ZONES ===")
        for name, data in world_state["detected_zones"].items():
            px = data.get("pixel", [0, 0])
            pos = data["position"]
            print(f"{name:12s}: Pixel=({px[0]:.1f}, {px[1]:.1f}) -> Est Pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        rclpy.shutdown()

    sub = node.create_subscription(Image, "/camera/image_raw", img_cb, 10)
    print("Waiting for /camera/image_raw...")
    try:
        rclpy.spin(node)
    except Exception:
        pass

if __name__ == "__main__":
    main()
