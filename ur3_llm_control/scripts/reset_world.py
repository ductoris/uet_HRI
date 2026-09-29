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

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import DeleteEntity


# Danh sách entity cần xóa khi reset
ENTITIES_TO_DELETE = [
    "red_cube",
    "yellow_cube",
    "blue_cube",
    "work_table",
    "zone_a_marker",
    "zone_b_marker",
    "zone_c_marker",
    "temp_zone_marker",
]


class WorldResetNode(Node):
    def __init__(self):
        super().__init__("world_reset_node")
        self._client = self.create_client(DeleteEntity, "/world/default/remove")

    def delete_entity(self, name: str) -> bool:
        """Xóa một entity khỏi Gazebo. Trả về True nếu thành công."""
        if not self._client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warning(
                "Service /world/default/remove không sẵn sàng – bỏ qua."
            )
            return False

        req = DeleteEntity.Request()
        req.name = name

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info(f"  ✓ Đã xóa entity: '{name}'")
            return True
        else:
            # Không có entity → không phải lỗi
            self.get_logger().debug(f"  - Entity '{name}' không tồn tại (bỏ qua).")
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
    print(f"✅ Đã xóa {deleted}/{len(ENTITIES_TO_DELETE)} entity.")
    print("   Sẵn sàng để spawn lại với scene_spawner.\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
