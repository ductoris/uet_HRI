#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_world.py — Xóa toàn bộ entity đã spawn trước đó khỏi Gazebo Ignition.

Chạy script này TRƯỚC scene_spawner để đảm bảo môi trường sạch,
tránh lỗi "Visual: [xxx] already exists".

Sử dụng:
    ros2 run ur3_llm_control reset_world

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import subprocess
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import DeleteEntity


ENTITIES_TO_DELETE = [
    "red_cube",
    "yellow_cube",
    "blue_cube",
    "work_table",
    "zone_a",
    "zone_b",
    "zone_c",
    "temp_zone",
]


class WorldResetNode(Node):
    def __init__(self):
        super().__init__("world_reset_node")
        self._clients = [
            self.create_client(DeleteEntity, "/world/empty/remove"),
            self.create_client(DeleteEntity, "/world/default/remove"),
        ]

    def delete_entity(self, name: str) -> bool:
        """Xóa entity qua ROS service hoặc ign service cli."""
        # 1. Thử ROS service
        for client in self._clients:
            if client.service_is_ready() or client.wait_for_service(timeout_sec=0.5):
                req = DeleteEntity.Request()
                req.name = name
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.5)
                if future.result() is not None and future.result().success:
                    self.get_logger().info(f"  ✓ Đã xóa entity: '{name}'")
                    return True

        # 2. Thử gọi ign CLI
        for world in ("empty", "default"):
            try:
                cmd = [
                    "ign", "service", "-s", f"/world/{world}/remove",
                    "--reqtype", "ignition.msgs.Entity",
                    "--reptype", "ignition.msgs.Boolean",
                    "--timeout", "800",
                    "--req", f'name: "{name}", type: MODEL',
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
                if "data: true" in res.stdout:
                    self.get_logger().info(f"  ✓ Đã xóa entity (CLI): '{name}'")
                    return True
            except Exception:
                pass

        return False


def main(args=None):
    rclpy.init(args=args)
    node = WorldResetNode()

    print("\n🔄 Đang reset Gazebo world...")
    print("=" * 50)

    deleted = 0
    for entity in ENTITIES_TO_DELETE:
        if node.delete_entity(entity):
            deleted += 1

    print("=" * 50)
    print(f"✅ Đã dọn dẹp {deleted} entity.")
    print("   Sẵn sàng để spawn lại với scene_spawner.\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
