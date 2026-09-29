#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_planner.py — Kết nối 9Router API (OpenAI-compatible) để sinh plan JSON.

Chức năng:
  1. Đọc biến môi trường LLM_API_KEY / NINE_ROUTER_API_KEY / OPENAI_API_KEY.
  2. Đọc LLM_BASE_URL / NINE_ROUTER_BASE_URL / OPENAI_BASE_URL.
  3. Đọc LLM_MODEL (mặc định: oc/muse-spark-1.2-contributor-free).
  4. Xây dựng System Prompt chặt chẽ kèm few-shot VI+EN.
  5. Gửi user command → nhận JSON plan.
  6. Có cơ chế retry nếu phản hồi bị lỗi JSON parse.
  7. Inject world_state vào prompt để LLM biết trạng thái scene.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import json
import time
from typing import Optional, Tuple

import yaml
try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    def get_package_share_directory(pkg_name: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Thư viện openai (tương thích 9Router)
from openai import OpenAI


# ---------------------------------------------------------------------------
#  Hằng số
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.0
DEFAULT_MODEL = "oc/muse-spark-1.2-contributor-free"
DEFAULT_BASE_URL = "http://localhost:20128/v1"


# ---------------------------------------------------------------------------
#  System Prompt
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

Example 1 (English):
User: "Please put the red cube in zone B."
Assistant:
[{"skill": "pick", "args": {"object": "red_cube"}}, {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}}, {"skill": "home", "args": {}}]

Example 2 (Vietnamese):
User: "Đưa khối màu vàng vào vùng C."
Assistant:
[{"skill": "pick", "args": {"object": "yellow_cube"}}, {"skill": "place", "args": {"object": "yellow_cube", "zone": "zone_c"}}, {"skill": "home", "args": {}}]

Example 3 (Vietnamese - alternate phrasing):
User: "Hãy lấy khối màu xanh và đặt nó vào ô A."
Assistant:
[{"skill": "pick", "args": {"object": "blue_cube"}}, {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}}, {"skill": "home", "args": {}}]

Example 4 (multi-step with temp_zone conflict):
User: "Arrange all objects according to my student ID."
(Current state: zone_a has blue_cube already, zone_b has red_cube, zone_c has yellow_cube — no conflict)
Assistant:
[{"skill": "home", "args": {}}]

Example 5 (multi-step with conflict):
User: "Arrange all objects according to my student ID."
(Current state: blue_cube at table, red_cube at zone_a, yellow_cube at table)
(Target: zone_a=blue_cube, zone_b=red_cube, zone_c=yellow_cube)
Assistant:
[{"skill": "pick", "args": {"object": "red_cube"}}, {"skill": "place", "args": {"object": "red_cube", "zone": "temp_zone"}}, {"skill": "pick", "args": {"object": "blue_cube"}}, {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}}, {"skill": "pick", "args": {"object": "red_cube"}}, {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}}, {"skill": "pick", "args": {"object": "yellow_cube"}}, {"skill": "place", "args": {"object": "yellow_cube", "zone": "zone_c"}}, {"skill": "home", "args": {}}]

Example 6 (invalid/unrelated command):
User: "What is the weather today?"
Assistant:
[]
"""


def _build_system_prompt(scene_cfg: dict, student_cfg: dict,
                          world_state: dict = None) -> str:
    """
    Xây dựng System Prompt đầy đủ cho LLM, kèm few-shot VI+EN.

    Parameters
    ----------
    scene_cfg : dict  — Cấu hình scene từ scene.yaml
    student_cfg : dict — Cấu hình sinh viên từ student_config.yaml
    world_state : dict — Trạng thái hiện tại {object_name: location}
    """
    student = student_cfg["student"]
    target = student_cfg["target_assignment"]

    obj_info = ""
    for obj_name, obj_data in scene_cfg["objects"].items():
        pos = obj_data["position"]
        obj_info += f"  - {obj_name}: initial position (x={pos['x']}, y={pos['y']})\n"

    zone_info = ""
    for zone_name, zone_data in scene_cfg["zones"].items():
        label = zone_data["label"]
        zone_info += f"  - {zone_name} ({label})\n"

    # Trạng thái hiện tại (inject world state)
    if world_state:
        state_lines = []
        for obj, loc in world_state.items():
            state_lines.append(f"  - {obj}: currently at {loc}")
        state_block = "=== CURRENT WORLD STATE ===\n" + "\n".join(state_lines) + "\n"
    else:
        state_block = ""

    system_prompt = f"""You are a Task Planner for a UR3e robotic arm in a tabletop pick-and-place environment.
Return ONLY a JSON array. Do not generate robot joint commands or trajectories.

=== ENVIRONMENT ===
Objects (cubes):
{obj_info}
Zones:
{zone_info}
{state_block}
=== AVAILABLE SKILLS (whitelist — use ONLY these) ===
- pick:         {{"skill": "pick", "args": {{"object": "<name>"}}}}
- place:        {{"skill": "place", "args": {{"object": "<name>", "zone": "<zone>"}}}}
- home:         {{"skill": "home", "args": {{}}}}
- move_above:   {{"skill": "move_above", "args": {{"target": "<name>"}}}}
- open_gripper: {{"skill": "open_gripper", "args": {{}}}}
- close_gripper:{{"skill": "close_gripper", "args": {{}}}}

Allowed objects: ["red_cube", "yellow_cube", "blue_cube"]
Allowed zones:   ["zone_a", "zone_b", "zone_c", "temp_zone"]

=== STUDENT ASSIGNMENT ===
Student: {student['name']} | MSSV: {student['mssv']} | P = {student['P']}
Target ("arrange by student ID"):
  zone_a ← {target['zone_a']}
  zone_b ← {target['zone_b']}
  zone_c ← {target['zone_c']}
Use temp_zone to break placement cycles.

=== HARD RULES ===
1. Output: JSON array ONLY — no markdown, no code blocks, no explanation.
2. Always pick before place. Never place without holding the object.
3. Always end the plan with {{"skill": "home", "args": {{}}}}.
4. Never use skills outside the whitelist.
5. If command is unrelated or impossible, return empty array: []
6. NEVER generate joint positions, trajectories, or coordinates.
{_FEW_SHOT_EXAMPLES}"""

    return system_prompt


# ---------------------------------------------------------------------------
#  Lớp LLMPlanner
# ---------------------------------------------------------------------------
class LLMPlanner:
    """
    Giao tiếp với 9Router (OpenAI-compatible API) để sinh plan JSON.
    API key và base URL đọc từ biến môi trường — KHÔNG hard-code.
    """

    def __init__(self, model: str = None):
        """
        Parameters
        ----------
        model : str | None
            Tên model. Nếu None → đọc LLM_MODEL từ env, fallback DEFAULT_MODEL.
        """
        # --- Đọc biến môi trường (không hard-code) ---
        api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("NINE_ROUTER_API_KEY")
            or os.environ.get("NINEROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("NINE_ROUTER_BASE_URL")
            or os.environ.get("ROBOT_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        )
        model_name = (
            model
            or os.environ.get("LLM_MODEL")
            or os.environ.get("ROBOT_LLM_MODEL")
            or DEFAULT_MODEL
        )

        if not api_key:
            print("[LLMPlanner] ⚠ Không tìm thấy API key trong môi trường. "
                  "Kích hoạt Smart Fallback Planner.")

        self._client = None
        if api_key:
            try:
                self._client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"[LLMPlanner] Kết nối tới {base_url} model={model_name}")
            except Exception as e:
                print(f"[LLMPlanner] Cảnh báo khởi tạo OpenAI client: {e}")

        self._model = model_name

        # Nạp config từ package share
        pkg_dir = get_package_share_directory("ur3_llm_control")
        with open(os.path.join(pkg_dir, "config", "scene.yaml"), "r") as f:
            self._scene_cfg = yaml.safe_load(f)
        with open(os.path.join(pkg_dir, "config", "student_config.yaml"), "r") as f:
            self._student_cfg = yaml.safe_load(f)

        # World state: {object_name: location_str}
        self._world_state: dict = {}
        self._system_prompt = _build_system_prompt(
            self._scene_cfg, self._student_cfg, self._world_state
        )

    def update_world_state(self, world_state: dict):
        """Cập nhật trạng thái scene để inject vào prompt."""
        self._world_state = dict(world_state)
        self._system_prompt = _build_system_prompt(
            self._scene_cfg, self._student_cfg, self._world_state
        )

    def _fallback_plan(self, user_command: str, world_state: dict = None) -> Optional[list]:
        """
        Rule-based Fallback Planner khi LLM API không phản hồi hoặc gặp sự cố mạng/500/429.
        Đảm bảo robot luôn có thể thực thi chính xác yêu cầu của đề bài, tự giải quyết xung đột bằng temp_zone.
        """
        cmd_lower = user_command.lower().strip()
        target = self._student_cfg.get("target_assignment", {})

        if world_state is None:
            world_state = self._world_state

        # 1. Yêu cầu sắp xếp theo MSSV / Đề bài
        student_triggers = [
            "student", "mssv", "đề bài", "de bai", "arrange", "sắp xếp", "sap xep",
            "theo mssv", "theo đề", "23020776", "bài tập", "bai tap", "tất cả", "tat ca", "all"
        ]
        if any(k in cmd_lower for k in student_triggers):
            # Với MSSV 23020776 -> P = 76 mod 6 = 4:
            # Zone A = blue_cube, Zone B = red_cube, Zone C = yellow_cube
            targets = {}
            for zone_key in ["zone_a", "zone_b", "zone_c"]:
                obj = target.get(zone_key)
                if obj:
                    targets[obj] = zone_key
            if not targets:
                targets = {"blue_cube": "zone_a", "red_cube": "zone_b", "yellow_cube": "zone_c"}
            return resolve_conflicts_and_generate_plan(targets, world_state)

        # 2. Lệnh về Home
        if any(k in cmd_lower for k in ["home", "về vị trí", "ve vi tri", "về nhà", "ve nha"]):
            return [{"skill": "home", "args": {}}]

        # 3. Mở/đóng gripper đơn lẻ
        if "open gripper" in cmd_lower or "mở kẹp" in cmd_lower or "mo kep" in cmd_lower:
            return [{"skill": "open_gripper", "args": {}}]
        if "close gripper" in cmd_lower or "đóng kẹp" in cmd_lower or "dong kep" in cmd_lower:
            return [{"skill": "close_gripper", "args": {}}]

        # 4. Nhận diện đối tượng (cube)
        obj = None
        if "blue" in cmd_lower or "xanh dương" in cmd_lower or "xanh" in cmd_lower:
            obj = "blue_cube"
        elif "red" in cmd_lower or "đỏ" in cmd_lower or "do" in cmd_lower:
            obj = "red_cube"
        elif "yellow" in cmd_lower or "vàng" in cmd_lower or "vang" in cmd_lower:
            obj = "yellow_cube"

        # 5. Nhận diện khu vực (zone)
        zone = None
        if any(k in cmd_lower for k in ["zone a", "zone_a", "khay a", "ô a", "o a", "to a", "vào a", "vào ô a", "vào khay a", " zone a"]):
            zone = "zone_a"
        elif any(k in cmd_lower for k in ["zone b", "zone_b", "khay b", "ô b", "o b", "to b", "vào b", "vào ô b", "vào khay b", " zone b"]):
            zone = "zone_b"
        elif any(k in cmd_lower for k in ["zone c", "zone_c", "khay c", "ô c", "o c", "to c", "vào c", "vào ô c", "vào khay c", " zone c"]):
            zone = "zone_c"
        elif any(k in cmd_lower for k in ["temp", "tạm", "tam", "khay phụ", "temp_zone", "buffer"]):
            zone = "temp_zone"
        else:
            # Fallback regex for single letter A/B/C at end of string or after zone/ô/vùng
            if " a" in cmd_lower or "a" == cmd_lower[-1:]:
                zone = "zone_a"
            elif " b" in cmd_lower or "b" == cmd_lower[-1:]:
                zone = "zone_b"
            elif " c" in cmd_lower or "c" == cmd_lower[-1:]:
                zone = "zone_c"

        # 6. Ghép thành skill plan hợp lệ có xử lý xung đột
        if obj and zone:
            targets = {obj: zone}
            return resolve_conflicts_and_generate_plan(targets, world_state)
        elif obj:
            return [
                {"skill": "pick", "args": {"object": obj}},
                {"skill": "home", "args": {}}
            ]

        return None

    def solve_conflict_plan(self, steps: list, user_command: str, world_state: dict = None) -> list:
        """
        Tự động gỡ xung đột cho một plan bị từ chối bằng cách tính toán di chuyển vào temp_zone.
        """
        if world_state is None:
            world_state = self._world_state

        targets = {}
        if steps:
            for step in steps:
                if step.get("skill") == "place":
                    args = step.get("args", {})
                    o = args.get("object")
                    z = args.get("zone")
                    if o and z and z in ["zone_a", "zone_b", "zone_c"]:
                        targets[o] = z

        if not targets:
            cmd_lower = user_command.lower().strip()
            student_triggers = ["student", "mssv", "đề bài", "arrange", "sắp xếp", "sap xep", "all", "tất cả"]
            if any(k in cmd_lower for k in student_triggers):
                target_cfg = self._student_cfg.get("target_assignment", {})
                for z_key in ["zone_a", "zone_b", "zone_c"]:
                    o = target_cfg.get(z_key)
                    if o:
                        targets[o] = z_key

        if targets:
            return resolve_conflicts_and_generate_plan(targets, world_state)

        return steps

    def plan(self, user_command: str,
             world_state: dict = None) -> Tuple[Optional[list], Optional[str]]:
        """
        Gửi lệnh người dùng tới LLM và nhận về plan JSON.
        Nếu API lỗi → tự động kích hoạt Rule-based Fallback Planner.

        Parameters
        ----------
        user_command : str  — Lệnh tự nhiên của người dùng
        world_state  : dict — Trạng thái scene hiện tại {obj: location}
        """
        # Cập nhật world state vào prompt trước khi gọi LLM
        if world_state is not None:
            self.update_world_state(world_state)

        last_error = None

        if self._client is not None:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": self._system_prompt},
                            {"role": "user", "content": user_command},
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                    )

                    raw_text = response.choices[0].message.content.strip()

                    # Parse JSON (loại bỏ markdown nếu có)
                    from ur3_llm_control.task_validator import parse_llm_response
                    steps, parse_error = parse_llm_response(raw_text)

                    if steps is not None:
                        print(f"[LLMPlanner] ✓ Plan nhận từ LLM ({len(steps)} bước).")
                        return steps, None
                    else:
                        last_error = f"Attempt {attempt}/{MAX_RETRIES}: {parse_error}"
                        print(f"[LLMPlanner] {last_error}")
                        if attempt < MAX_RETRIES:
                            # Gửi lại với thông báo lỗi để LLM sửa
                            corrective_msg = (
                                f"{user_command}\n\n"
                                f"[SYSTEM: Previous response had JSON error: {parse_error}. "
                                f"Return ONLY a valid JSON array.]"
                            )
                            user_command_retry = corrective_msg
                            time.sleep(RETRY_DELAY_SEC)

                except Exception as e:
                    last_error = f"Attempt {attempt}/{MAX_RETRIES}: API error: {e}"
                    print(f"[LLMPlanner] {last_error}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_SEC)
        else:
            last_error = "OpenAI client chưa được khởi tạo (thiếu API key)"

        # KÍCH HOẠT FALLBACK PLANNER
        fallback_steps = self._fallback_plan(user_command, world_state=world_state)
        if fallback_steps is not None:
            print(f"[LLMPlanner] ℹ Smart Fallback Planner kích hoạt: '{user_command}'")
            return fallback_steps, None

        return None, f"Thất bại sau {MAX_RETRIES} lần thử. Lỗi cuối: {last_error}"

    def get_student_info(self) -> dict:
        """Trả về thông tin sinh viên."""
        return self._student_cfg["student"]

    def get_target_assignment(self) -> dict:
        """Trả về ánh xạ mục tiêu."""
        return self._student_cfg["target_assignment"]


def resolve_conflicts_and_generate_plan(targets: dict, current_world_state: dict = None) -> list:
    """
    Tạo plan JSON tự động giải quyết xung đột bằng temp_zone khi khay mục tiêu đang bị vật khác chiếm.
    
    Parameters
    ----------
    targets : dict
        Dict ánh xạ {object_name: target_zone}, ví dụ:
        {"blue_cube": "zone_a", "red_cube": "zone_b", "yellow_cube": "zone_c"}
    current_world_state : dict
        Trạng thái vị trí hiện tại của các vật, ví dụ:
        {"yellow_cube": "zone_a", "blue_cube": "table", "red_cube": "table"}
    """
    if current_world_state is None:
        current_world_state = {}

    allowed_objs = ["red_cube", "yellow_cube", "blue_cube"]
    allowed_zones = ["zone_a", "zone_b", "zone_c", "temp_zone"]

    curr_loc = {obj: current_world_state.get(obj, "table") for obj in allowed_objs}
    zone_occ = {z: None for z in allowed_zones}
    for obj, loc in curr_loc.items():
        if loc in allowed_zones:
            zone_occ[loc] = obj

    steps = []
    max_steps = 30
    iterations = 0

    while iterations < max_steps:
        iterations += 1
        unmet = [obj for obj, t_zone in targets.items() if curr_loc[obj] != t_zone]
        if not unmet:
            break

        moved_any = False

        # 1. Thử di chuyển vật mà zone mục tiêu của nó đang trống
        for obj in unmet:
            t_zone = targets[obj]
            occupant = zone_occ.get(t_zone)
            if occupant is None or occupant == obj:
                old_loc = curr_loc[obj]
                if old_loc in zone_occ and zone_occ[old_loc] == obj:
                    zone_occ[old_loc] = None

                steps.append({"skill": "pick", "args": {"object": obj}})
                steps.append({"skill": "place", "args": {"object": obj, "zone": t_zone}})

                curr_loc[obj] = t_zone
                zone_occ[t_zone] = obj
                moved_any = True
                break

        if moved_any:
            continue

        # 2. Phát hiện xung đột / chu kỳ: tìm vật đang chiếm zone mục tiêu của vật khác
        blocking_obj = None
        for obj in unmet:
            t_zone = targets[obj]
            occ = zone_occ.get(t_zone)
            if occ is not None and occ != obj and curr_loc[occ] != "temp_zone":
                blocking_obj = occ
                break

        if blocking_obj is None:
            for z in ["zone_a", "zone_b", "zone_c"]:
                occ = zone_occ.get(z)
                if occ is not None and targets.get(occ) != z and curr_loc[occ] != "temp_zone":
                    blocking_obj = occ
                    break

        if blocking_obj is not None:
            old_loc = curr_loc[blocking_obj]
            if old_loc in zone_occ and zone_occ[old_loc] == blocking_obj:
                zone_occ[old_loc] = None

            steps.append({"skill": "pick", "args": {"object": blocking_obj}})
            steps.append({"skill": "place", "args": {"object": blocking_obj, "zone": "temp_zone"}})

            curr_loc[blocking_obj] = "temp_zone"
            zone_occ["temp_zone"] = blocking_obj
            moved_any = True
        else:
            break

    if not steps or steps[-1].get("skill") != "home":
        steps.append({"skill": "home", "args": {}})

    return steps

