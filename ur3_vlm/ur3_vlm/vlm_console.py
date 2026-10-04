#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_console.py — Bàn điều khiển tích hợp toàn diện VLM Robot UR3e (Giai đoạn 5).

Quy trình hoạt động End-to-End:
  1. Nhận lệnh ngôn ngữ tự nhiên từ người dùng (Tiếng Việt / Tiếng Anh).
  2. Lấy trạng thái thế giới (WorldState JSON) thời gian thực từ Camera Perception.
  3. Gửi Prompt đa phương thức / JSON context sang VLM/LLM Planner (9Router / Gemini / Offline).
  4. Trích xuất danh sách kỹ năng thao tác (skills plan: pick, place, stack, unstack, home).
  5. Điều phối RobotSkills thực thi mượt mà với kẹp vật lý DetachableJoint trong Gazebo.
  6. Báo cáo trạng thái trực quan và cập nhật kết quả.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import sys
import time
import json
import threading
from typing import Dict, List, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

from ur3_vlm.camera_perception import CameraPerceptionNode
from ur3_vlm.llm_planner import LLMPlanner
from ur3_vlm.robot_skills import RobotSkills, SkillResult


class VLMConsoleNode(Node):
    """Node tích hợp toàn diện Perception + LLM + Skills."""

    def __init__(self):
        super().__init__("vlm_console_node")
        self.get_logger().info("=== Khởi tạo VLM Robot Console (Bài tập tuần 3 - Mai Đức Trí) ===")

        # Khởi tạo Planner & Skills
        self.planner = LLMPlanner()
        self.skills = RobotSkills(self)

        # Trạng thái nhận diện từ camera (QoS Transient Local)
        self.world_state: Dict[str, Any] = {}
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )
        self._world_state_sub = self.create_subscription(
            String,
            "/world_state_json",
            self._world_state_callback,
            qos_state,
        )

        self.get_logger().info("VLM Console đã sẵn sàng!")

    def _world_state_callback(self, msg: String):
        try:
            self.world_state = json.loads(msg.data)
            if hasattr(self, 'skills') and self.skills:
                self.skills._world_state_data = self.world_state
                self.skills._detected_objects = self.world_state.get("detected_objects", {})
                self.skills._detected_zones = self.world_state.get("detected_zones", {})
        except Exception:
            pass

    def get_latest_world_state(self) -> dict:
        return self.world_state

    def execute_plan(self, plan_actions: List[dict]) -> bool:
        """Thực thi tuần tự danh sách action từ LLM."""
        if not plan_actions:
            self.get_logger().warn("Kế hoạch rỗng, không có hành động nào để thực hiện.")
            return False

        print("\n" + "=" * 65)
        print(f"  BẮT ĐẦU THỰC THI KẾ HOẠCH ({len(plan_actions)} BƯỚC)")
        print("=" * 65)

        for idx, action in enumerate(plan_actions, start=1):
            skill = action.get("skill", "").lower()
            args = action.get("args", {})
            desc = action.get("description", "")

            print(f"\n[Bước {idx}/{len(plan_actions)}] > Skill: {skill.upper()} | {desc}")
            print(f"               Tham số: {args}")

            res = SkillResult.FAILED

            if skill == "home":
                res = self.skills.home()

            elif skill == "open_gripper":
                self.skills.open_gripper()
                res = SkillResult.SUCCESS

            elif skill == "close_gripper":
                obj = args.get("object") or args.get("object_name")
                self.skills.close_gripper(obj)
                res = SkillResult.SUCCESS

            elif skill == "pick":
                obj = args.get("object") or args.get("object_name")
                if not obj:
                    print("  ❌ Thiếu tham số 'object'")
                    return False
                res = self.skills.pick(obj)

            elif skill == "place":
                obj = args.get("object") or args.get("object_name")
                target = args.get("target") or args.get("zone")
                if not obj or not target:
                    print("  ❌ Thiếu tham số 'object' hoặc 'target'")
                    return False

                # ── Kiểm tra an toàn Camera Perception trước khi Place ──
                # Nếu vùng đích là Zone và Camera phát hiện vẫn còn khối khác cản trở (do lần dọn trước gắp trượt), tự động dọn lại lần 2
                target_zone_data = self.world_state.get("detected_zones", {}).get(target, {})
                occupant = target_zone_data.get("occupant")
                if occupant and occupant != obj:
                    print(f"  ⚠️ [Camera Perception]: Ô '{target}' vẫn đang bị chiếm bởi '{occupant}' (lần dọn trước có thể bị trượt)!")
                    print(f"  🔄 Đang tự động dọn khối cản trở '{occupant}' sang vùng tạm [T] lần 2 trước khi đặt...")
                    
                    # Nếu đang giữ obj trong kẹp, tạm đặt xuống bàn/vùng tạm
                    if self.skills._attached_object == obj:
                        self.skills.place(obj, "staging_area")
                    
                    # Dọn khối cản trở
                    self.skills.pick(occupant)
                    self.skills.place(occupant, "staging_area")
                    
                    # Gắp lại obj chính nếu cần
                    if self.skills._attached_object != obj:
                        self.skills.pick(obj)

                res = self.skills.place(obj, target)

            elif skill == "stack":
                bot = args.get("bottom_object") or args.get("bottom")
                top = args.get("top_object") or args.get("top")
                if not bot or not top:
                    print("  ❌ Thiếu tham số 'bottom_object' hoặc 'top_object'")
                    return False
                res = self.skills.stack(bot, top)

            elif skill == "unstack":
                top = args.get("top_object") or args.get("top")
                target = args.get("target") or args.get("zone")
                if not top or not target:
                    print("  ❌ Thiếu tham số 'top_object' hoặc 'target'")
                    return False
                res = self.skills.unstack(top, target)

            elif skill == "move_to":
                target = args.get("target") or args.get("object") or args.get("zone")
                if not target:
                    print("  ❌ Thiếu tham số 'target'")
                    return False
                res = self.skills.move_to(target)

            else:
                print(f"  ❌ Skill không hỗ trợ: {skill}")
                return False

            if res != SkillResult.SUCCESS:
                print(f"  ❌ Bước {idx} thất bại với kết quả: {res.value}")
                return False

            print(f"  ✅ Bước {idx} hoàn thành thành công!")
            time.sleep(0.5)

        print("\n" + "=" * 65)
        print("  🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC BƯỚC TRONG KẾ HOẠCH!")
        print("=" * 65 + "\n")
        return True


def print_banner():
    print(r"""
======================================================================
  _   _ ____  _____        __     ___     __  __ 
 | | | |  _ \|___ / ___    \ \   / / |   |  \/  |
 | | | | |_) | |_ \/ _ \    \ \ / /| |   | |\/| |
 | |_| |  _ < ___) \___/     \ V / | |___| |  | |
  \___/|_| \_\____/           \_/  |_____|_|  |_|
  
  HỆ THỐNG ĐIỀU KHIỂN ROBOT UR3e BẰNG VISION-LANGUAGE MODEL (BÀI 03)
  Sinh viên thực hiện: Mai Đức Trí | MSSV: 23020776
======================================================================
""")


def print_help():
    print("""
Các lệnh có thể sử dụng:
  • Nhập câu lệnh tự nhiên (Tiếng Việt/Tiếng Anh), ví dụ:
      - "hãy gắp khối màu đỏ vào ô b"
      - "chuyển khối xanh lá vào zone_a và khối vàng vào zone_c"
      - "xếp khối đỏ lên trên khối xanh dương"
      - "dọn dẹp tất cả các khối vào các ô trống"
  • Lệnh hệ thống đặc biệt:
      - /state     : Xem trạng thái nhận diện Camera Perception hiện tại
      - /home      : Đưa robot về vị trí Home an toàn
      - /open      : Mở kẹp
      - /close <c> : Đóng kẹp (ví dụ: /close red_cube)
      - /help      : Hiển thị lại hướng dẫn
      - /exit      : Thoát chương trình
""")


def main(args=None):
    rclpy.init(args=args)
    console_node = VLMConsoleNode()

    # Background spin thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(console_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(1.0)
    print_banner()
    print_help()

    try:
        while rclpy.ok():
            try:
                user_input = input("\n[VLM User] > ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", "q"]:
                break

            elif user_input.lower() == "/help":
                print_help()

            elif user_input.lower() == "/home":
                console_node.skills.home()

            elif user_input.lower() == "/open":
                console_node.skills.open_gripper()

            elif user_input.lower().startswith("/close"):
                parts = user_input.split()
                obj = parts[1] if len(parts) > 1 else None
                console_node.skills.close_gripper(obj)

            elif user_input.lower() == "/state":
                state = console_node.get_latest_world_state()
                print("\n[Trạng thái WorldState hiện tại]:")
                print(json.dumps(state, indent=2, ensure_ascii=False))

            else:
                # 1. Thu thập WorldState
                state = console_node.get_latest_world_state()
                if not state:
                    print("⚠️ Chưa nhận được dữ liệu từ /world_state_json. Đang thử lập kế hoạch với Scene mặc định...")

                # 2. Lập kế hoạch qua VLM Planner
                print(f"\n🧠 [VLM Planner] Đang suy luận và lập kế hoạch cho: '{user_input}'...")
                plan_result = console_node.planner.plan(user_input, state)

                actions = plan_result if isinstance(plan_result, list) else plan_result.get("plan", [])
                reasoning = plan_result.get("reasoning", "") if isinstance(plan_result, dict) else f"Kế hoạch thực thi gồm {len(actions)} bước."
                print(f"📋 [Lý giải kế hoạch]: {reasoning}")

                if not actions:
                    print("⚠️ Không tìm thấy hoặc LLM không sinh ra hành động nào hợp lệ cho câu lệnh này.")
                    continue

                # 3. Thực thi kế hoạch
                console_node.execute_plan(actions)

    except KeyboardInterrupt:
        pass
    finally:
        console_node.skills.shutdown()
        console_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
