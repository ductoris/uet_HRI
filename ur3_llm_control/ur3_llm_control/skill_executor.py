#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skill_executor.py — ROS 2 Main Node: điều phối luồng LLM → Validate → Execute.

Chức năng:
  1. Nhận lệnh người dùng từ terminal (interactive console).
  2. Gửi lệnh tới LLMPlanner → nhận plan JSON (kèm world state).
  3. Validate plan qua TaskValidator (kèm world state).
  4. Thực thi từng skill qua RobotSkills.
  5. Theo dõi world state (vật nào ở zone nào) và cập nhật sau mỗi bước.
  6. In terminal đúng chuẩn định dạng yêu cầu đề bài.

Chế độ:
  --dry-run : Chỉ in plan, KHÔNG thực thi robot.
  --mock-llm: Dùng Mock LLM để test (không cần mạng/API).

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import sys
import threading
import argparse

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ur3_llm_control.llm_planner import LLMPlanner
from ur3_llm_control.task_validator import (
    parse_llm_response,
    validate_plan,
    format_plan_text,
    ALLOWED_OBJECTS,
    ALLOWED_ZONES,
)
from ur3_llm_control.robot_skills import RobotSkills, SkillResult


# ---------------------------------------------------------------------------
#  Hằng số hiển thị
# ---------------------------------------------------------------------------
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         UR3e LLM Skill-based Planning Controller               ║
║         Sinh viên: Mai Duc Tri | MSSV: 23020776                ║
║         P = 4: Zone A=blue, Zone B=red, Zone C=yellow          ║
╠══════════════════════════════════════════════════════════════════╣
║  Lệnh ví dụ:                                                   ║
║    • Please put the red cube in zone B.                         ║
║    • Arrange all objects according to my student ID.            ║
║    • Đưa khối màu đỏ vào vùng B.                               ║
║    • Hãy lấy khối màu vàng và đặt nó vào ô A.                  ║
║  Gõ 'quit' hoặc 'exit' để thoát.                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

SEPARATOR = "=" * 64


# ---------------------------------------------------------------------------
#  Hàm format kết quả thực thi
# ---------------------------------------------------------------------------
def _format_skill_call(step: dict) -> str:
    """Chuyển step dict thành chuỗi hàm dạng readable."""
    skill = step.get("skill", "?")
    args = step.get("args", {})

    if skill == "pick":
        return f"pick({args.get('object', '?')})"
    elif skill == "place":
        return f"place({args.get('object', '?')}, {args.get('zone', '?')})"
    elif skill == "move_above":
        target = args.get("target") or args.get("object") or args.get("zone", "?")
        return f"move_above({target})"
    elif skill in ("home", "open_gripper", "close_gripper"):
        return f"{skill}()"
    else:
        return f"{skill}({args})"


def _print_execution_line(skill_str: str, result: SkillResult):
    """In dòng kết quả thực thi với padding dots."""
    max_width = 40
    dots_count = max(1, max_width - len(skill_str))
    dots = "." * dots_count
    print(f"{skill_str} {dots} {result.value}")


def _print_dry_run_line(skill_str: str):
    """In dòng dry-run với padding dots."""
    max_width = 40
    dots_count = max(1, max_width - len(skill_str))
    dots = "." * dots_count
    print(f"{skill_str} {dots} [DRY-RUN]")


# ---------------------------------------------------------------------------
#  World State Manager
# ---------------------------------------------------------------------------
class WorldStateManager:
    """
    Theo dõi vị trí của từng object sau mỗi skill.

    State format: {object_name: "table" | zone_name | "gripper"}
    """

    def __init__(self):
        # Mặc định: tất cả vật trên bàn
        self._state: dict = {obj: "table" for obj in ALLOWED_OBJECTS}
        self._holding: str = None

    @property
    def state(self) -> dict:
        return dict(self._state)

    @property
    def holding(self) -> str:
        return self._holding

    def apply_pick(self, obj: str):
        if obj in self._state:
            self._state[obj] = "gripper"
            self._holding = obj

    def apply_place(self, obj: str, zone: str):
        if obj in self._state:
            self._state[obj] = zone
            self._holding = None

    def apply_home(self):
        pass  # Home không thay đổi vị trí object

    def get_world_state_str(self) -> str:
        """Trả về chuỗi mô tả world state để in ra terminal."""
        lines = ["  [World State]"]
        for obj, loc in self._state.items():
            lines.append(f"    {obj}: {loc}")
        if self._holding:
            lines.append(f"    Gripper holding: {self._holding}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Node chính
# ---------------------------------------------------------------------------
class SkillExecutorNode(Node):
    """ROS 2 Node điều phối LLM planning và skill execution."""

    def __init__(self, dry_run: bool = False, mock_llm: bool = False):
        super().__init__("skill_executor_node")
        self._dry_run = dry_run
        self._mock_llm = mock_llm

        if dry_run:
            self.get_logger().info("*** CHẾ ĐỘ DRY-RUN: Chỉ in plan, không thực thi robot ***")
        if mock_llm:
            self.get_logger().info("*** CHẾ ĐỘ MOCK LLM: Không gọi API ***")

        self.get_logger().info("SkillExecutorNode đang khởi tạo...")

        # World state tracker
        self._world = WorldStateManager()

        # Khởi tạo RobotSkills (chỉ khi không dry-run)
        self._skills = None
        if not dry_run:
            self._skills = RobotSkills(self)

        # Khởi tạo LLM Planner
        try:
            self._planner = LLMPlanner()
            self.get_logger().info("LLMPlanner kết nối thành công.")
        except Exception as e:
            self.get_logger().error(f"LLMPlanner lỗi: {e}")
            self._planner = None

        # Cache last target for optimization
        self._last_above_target = None
        self.get_logger().info("SkillExecutorNode sẵn sàng.")

    def destroy_node(self):
        if self._skills is not None:
            self._skills.shutdown()
        return super().destroy_node()

    def execute_command(self, user_command: str) -> bool:
        """
        Xử lý một lệnh người dùng: LLM → Validate → Execute.

        Returns
        -------
        bool : True nếu toàn bộ task thành công.
        """
        print(f"\n{SEPARATOR}")
        print(f"USER COMMAND:\n{user_command}")
        print()

        # ---- Bước 1: Gọi LLM (hoặc mock) ----
        if self._planner is None:
            print("[ERROR] LLMPlanner chưa được khởi tạo.")
            return False

        if self._mock_llm:
            steps, error = self._planner._fallback_plan(user_command), None
            if steps is None:
                steps = []
            print("[MOCK-LLM] Dùng Fallback Planner (không gọi API)")
        else:
            print("[*] Đang gửi lệnh tới LLM...")
            steps, error = self._planner.plan(
                user_command,
                world_state=self._world.state
            )

        if steps is None:
            print(f"[ERROR] LLM thất bại: {error}")
            return False

        # Plan rỗng = lệnh không thực hiện được / không liên quan
        if len(steps) == 0:
            print("LLM PLAN:\n(plan rỗng — lệnh không liên quan hoặc không thực hiện được)")
            print()
            print("TASK SKIPPED (no action required)")
            print(SEPARATOR)
            return True

        # ---- Bước 2: In plan ----
        plan_text = format_plan_text(steps)
        print("LLM PLAN:")
        for line in plan_text.split("\n"):
            print(f"{line}")
        print()

        # ---- Bước 3: Validate ----
        is_valid, errors = validate_plan(
            steps,
            initial_world_state=self._world.state
        )
        if not is_valid:
            has_conflict = any("chiếm" in str(err).lower() or "temp_zone" in str(err).lower() or "đang bị" in str(err).lower() for err in errors)
            if has_conflict and self._planner is not None:
                print("[AUTOPLAN] Phát hiện xung đột zone. Tự động quy hoạch vùng tạm (temp_zone)...")
                fixed_steps = self._planner.solve_conflict_plan(
                    steps,
                    user_command,
                    world_state=self._world.state
                )
                if fixed_steps:
                    is_valid_fixed, errors_fixed = validate_plan(
                        fixed_steps,
                        initial_world_state=self._world.state
                    )
                    if is_valid_fixed:
                        steps = fixed_steps
                        is_valid = True
                        errors = []
                        print("[AUTOPLAN] ✓ Plan đã được tự động giải quyết xung đột thành công!")
                        print("LLM PLAN (FIXED):")
                        for line in format_plan_text(steps).split("\n"):
                            print(f"{line}")
                        print()

        if not is_valid:
            print("PLAN REJECTED:")
            for err in errors:
                print(f"  ✗ {err}")
            print()
            print("PLAN REJECTED — Không thực thi.")
            print(SEPARATOR)
            return False

        # ---- Bước 4: Dry-run hoặc Thực thi ----
        print("EXECUTION:")
        all_success = True

        for step in steps:
            skill = step.get("skill")
            args = step.get("args", {})
            skill_str = _format_skill_call(step)

            if self._dry_run:
                _print_dry_run_line(skill_str)
                self._update_world_state_dry(skill, args)
            else:
                result = self._dispatch_skill(skill, args)
                _print_execution_line(skill_str, result)

                if result != SkillResult.SUCCESS:
                    all_success = False
                    print(f"\n[!] Skill '{skill_str}' thất bại ({result.value}). Dừng thực thi.")
                    break

                # Cập nhật world state sau khi skill thành công
                self._update_world_state(skill, args)

        # ---- Bước 5: Kết quả tổng ----
        print()
        if self._dry_run:
            print("TASK DRY-RUN COMPLETE")
        elif all_success:
            print("TASK SUCCESS")
        else:
            print("TASK FAILED")
        print(SEPARATOR)

        return all_success

    def _update_world_state(self, skill: str, args: dict):
        """Cập nhật world state sau khi skill thực thi thành công."""
        if skill == "pick":
            obj = args.get("object", "")
            if obj:
                self._world.apply_pick(obj)
        elif skill == "place":
            obj = args.get("object", "")
            zone = args.get("zone", "")
            if obj and zone:
                self._world.apply_place(obj, zone)
        elif skill == "home":
            self._world.apply_home()

    def _update_world_state_dry(self, skill: str, args: dict):
        """Cập nhật world state cho dry-run."""
        self._update_world_state(skill, args)

    def _dispatch_skill(self, skill: str, args: dict) -> SkillResult:
        """Dispatch một skill tới RobotSkills."""
        if self._skills is None:
            self.get_logger().error("RobotSkills chưa khởi tạo (dry-run mode?)")
            return SkillResult.FAILED

        try:
            if skill == "home":
                return self._skills.home()
            elif skill == "pick":
                return self._skills.pick(args.get("object", ""))
            elif skill == "place":
                return self._skills.place(
                    args.get("object", ""),
                    args.get("zone", ""),
                )
            elif skill == "move_above":
                target = (
                    args.get("target")
                    or args.get("object")
                    or args.get("zone", "")
                )
                return self._skills.move_above(target)
            elif skill == "open_gripper":
                return self._skills.open_gripper()
            elif skill == "close_gripper":
                return self._skills.close_gripper()
            else:
                self.get_logger().error(f"Skill không xác định: {skill}")
                return SkillResult.FAILED

        except Exception as e:
            self.get_logger().error(f"Lỗi khi thực thi {skill}: {e}")
            return SkillResult.FAILED


# ---------------------------------------------------------------------------
#  Interactive console loop
# ---------------------------------------------------------------------------
def _interactive_loop(node: SkillExecutorNode):
    """Vòng lặp nhập lệnh từ terminal."""
    print(BANNER)

    if node._dry_run:
        print("*** DRY-RUN MODE: Robot sẽ KHÔNG di chuyển ***\n")
    if node._mock_llm:
        print("*** MOCK-LLM MODE: Dùng Fallback Planner (không cần API) ***\n")

    while rclpy.ok():
        try:
            user_input = input("\n🤖 Nhập lệnh (hoặc 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Thoát chương trình.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("[*] Thoát chương trình.")
            break

        if user_input.lower() == "state":
            print(node._world.get_world_state_str())
            continue

        node.execute_command(user_input)


# ---------------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------------
def main(args=None):
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="UR3e LLM Skill-based Planning Controller")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in plan, không thực thi robot"
    )
    parser.add_argument(
        "--mock-llm", action="store_true",
        help="Dùng Mock LLM (Fallback Planner), không cần API"
    )

    # Tách ROS args khỏi custom args
    import sys
    ros_args = []
    custom_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ("--dry-run", "--mock-llm"):
            custom_args.append(sys.argv[i])
            i += 1
        else:
            ros_args.append(sys.argv[i])
            i += 1

    parsed = parser.parse_args(custom_args)

    rclpy.init(args=args)

    node = SkillExecutorNode(
        dry_run=parsed.dry_run,
        mock_llm=parsed.mock_llm,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        _interactive_loop(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
