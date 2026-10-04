#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_planner.py — Trình lập kế hoạch hành động (LLM / VLM Task Planner) cho UR3e (Bài 03).

Chức năng:
  1. Nhận câu lệnh ngôn ngữ tự nhiên (Tiếng Việt / English) từ người dùng.
  2. Tích hợp trạng thái thế giới nhận diện từ Camera Perception (WorldState JSON).
  3. Hỗ trợ đa dạng Backend:
     - Google Gemini API (gemini-1.5-flash / gemini-2.0-flash / gemini-1.5-pro)
     - OpenAI-compatible API (9Router / OpenAI / Local LLM)
     - Offline Heuristic Planner Fallback: Bộ lập kế hoạch cục bộ thông minh,
       hoạt động 100% không cần internet / API key.
  4. Hỗ trợ đầy đủ bộ kỹ năng (Skill Set Bài 03):
     - pick(object)
     - place(object, target)
     - stack(bottom_object, top_object)
     - unstack(top_object, target)
     - move_to(target)
     - home()
  5. Cơ chế kiểm tra, sửa lỗi kế hoạch tự động (Plan Validation & Auto-Repair).

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Tuple, Optional

import yaml
try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    def get_package_share_directory(pkg_name: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Thư viện OpenAI nếu có sẵn
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ===========================================================================
#  Hằng số & Cấu hình mặc định
# ===========================================================================
#  Hằng số & Cấu hình mặc định
# ===========================================================================
ALLOWED_OBJECTS = ["red_cube", "yellow_cube", "blue_cube", "green_cube", "purple_cube"]
ALLOWED_ZONES = ["zone_a", "zone_b", "zone_c", "staging_area"]
ALLOWED_SKILLS = ["pick", "place", "stack", "unstack", "move_to", "home"]

COLOR_KEYWORDS = {
    "red_cube": ["đỏ", "màu đỏ", "đor", "dor", "do", "mau do", "red", "red cube", "red_cube"],
    "yellow_cube": ["vàng", "màu vàng", "vang", "vaangf", "vaang", "mau vang", "yellow", "yellow cube", "yellow_cube"],
    "blue_cube": ["xanh dương", "xanh duong", "xanh nước biển", "xanh lam", "blue", "blue cube", "blue_cube"],
    "green_cube": ["xanh lá", "xanh la", "xanh lục", "green", "green cube", "green_cube"],
    "purple_cube": ["tím", "màu tím", "tim", "tiems", "mau tim", "purple", "purple cube", "purple_cube"],
}

ZONE_KEYWORDS = {
    "zone_a": ["zone a", "zone_a", "vùng a", "vung a", "ô a", "o a", "khay a", "hộp a", "a"],
    "zone_b": ["zone b", "zone_b", "vùng b", "vung b", "ô b", "o b", "khay b", "hộp b", "b"],
    "zone_c": ["zone c", "zone_c", "vùng c", "vung c", "ô c", "o c", "khay c", "hộp c", "c"],
    "staging_area": [
        "vùng t", "vung t", "zone t", "zone_t", "ô t", "o t", "khay t", "hộp t", "t",
        "vùng tạm", "vung tam", "vùng đệm", "vung dem", "vung temp", "vùng temp", "temp", "bàn", "mặt bàn", "ra bàn", "ngoài", "ra ngoài", "staging"
    ],
}


# ===========================================================================
#  Bộ lập kế hoạch Heuristic Cục bộ (Offline Planner Fallback)
# ===========================================================================
class OfflineHeuristicPlanner:
    """Bộ lập kế hoạch ngữ nghĩa deterministic chạy offline 100% hỗ trợ đa dạng biến thể ngôn ngữ."""

    @staticmethod
    def extract_objects(text: str) -> List[str]:
        text_lower = text.lower()
        matches = []  # list of (start_idx, length, obj_name)
        
        # Duyệt qua các keyword, ưu tiên keyword dài trước
        for obj_name, kws in COLOR_KEYWORDS.items():
            for kw in kws:
                for m in re.finditer(r'(?:\b|_)' + re.escape(kw) + r'(?:\b|_)', text_lower):
                    matches.append((m.start(), m.end() - m.start(), obj_name))

        # Sắp xếp ưu tiên: vị trí start tăng dần, nếu trùng vị trí thì độ dài lớn hơn được ưu tiên
        matches.sort(key=lambda x: (x[0], -x[1]))

        # Lọc bỏ trùng lặp và các match con bị đè
        ordered_objs = []
        last_end = -1
        for start, length, obj_name in matches:
            if start >= last_end:
                if obj_name not in ordered_objs:
                    ordered_objs.append(obj_name)
                last_end = start + length

        return ordered_objs

    @staticmethod
    def extract_object(text: str) -> Optional[str]:
        objs = OfflineHeuristicPlanner.extract_objects(text)
        return objs[0] if objs else None

    @staticmethod
    def extract_zones(text: str) -> List[str]:
        text_lower = text.lower()
        matches = []
        for zone_name, kws in ZONE_KEYWORDS.items():
            for kw in kws:
                for m in re.finditer(r'(?:\b|_)' + re.escape(kw) + r'(?:\b|_)', text_lower):
                    matches.append((m.start(), m.end() - m.start(), zone_name))
        matches.sort(key=lambda x: (x[0], -x[1]))
        ordered_zones = []
        last_end = -1
        for start, length, zone_name in matches:
            if start >= last_end:
                if zone_name not in ordered_zones:
                    ordered_zones.append(zone_name)
                last_end = start + length
        return ordered_zones

    @staticmethod
    def extract_zone(text: str) -> Optional[str]:
        zones = OfflineHeuristicPlanner.extract_zones(text)
        return zones[0] if zones else None

    def plan(self, command: str, world_state: dict = None) -> List[dict]:
        """Phân tích cú pháp câu lệnh và sinh plan tương ứng."""
        cmd_lower = command.lower().strip()
        plan = []

        # 1. Lệnh Reset / Home / Safe Pose
        if any(w in cmd_lower for w in ["home", "về vị trí ban đầu", "ve vi tri ban dau", "về home", "ve home", "về nhà", "reset", "an toàn", "an toan", "safe", "về gốc", "gốc", "nghỉ", "park"]):
            return [{"skill": "home", "args": {}}]

        # 2. Lệnh Xếp chồng (Stacking): "đặt X lên trên Y", "xếp X lên trên Y", "stack X on Y"
        if any(w in cmd_lower for w in ["lên trên", "len tren", "treen", "tren", "chồng", "chong", "stack", "on top of"]):
            objs = self.extract_objects(cmd_lower)
            if len(objs) >= 2:
                # Trong tiếng Việt/Anh: "Đặt/Xếp khối A lên trên khối B" -> top=A, bottom=B
                top_obj, bottom_obj = objs[0], objs[1]
                plan.append({"skill": "pick", "args": {"object": top_obj}})
                plan.append({"skill": "stack", "args": {"bottom_object": bottom_obj, "top_object": top_obj}})
                plan.append({"skill": "home", "args": {}})
                return plan

        # 3. Lệnh Dỡ chồng (Unstack): "dỡ X ra", "unstack X to Y"
        if any(w in cmd_lower for w in ["dỡ", "dỡ ra", "do ra", "unstack", "lấy khối trên cùng", "lay khoi tren"]):
            obj = self.extract_object(cmd_lower)
            target_zone = self.extract_zone(cmd_lower) or "zone_a"
            if obj:
                plan.append({"skill": "unstack", "args": {"top_object": obj, "target": target_zone}})
                plan.append({"skill": "home", "args": {}})
                return plan

        # 4. Lệnh Gắp & Đặt (Pick and Place) - Hỗ trợ cả 1 hoặc nhiều khối liên tiếp
        objs = self.extract_objects(cmd_lower)
        zones = self.extract_zones(cmd_lower)

        if objs and zones:
            for idx, obj in enumerate(objs):
                target_zone = zones[idx] if idx < len(zones) else zones[-1]
                
                # Kiểm tra xem zone đích có bị chiếm dụng không
                occupied_by = None
                if world_state:
                    zone_info = world_state.get("detected_zones", {}).get(target_zone, {})
                    occupied_by = zone_info.get("occupant")

                # Nếu zone đích đang có khối khác (và không phải khối chuẩn bị gắp), dọn khối đó trước
                if occupied_by and occupied_by != obj:
                    free_zones = world_state.get("summary", {}).get("free_zones", [])
                    other_zones = [z for z in ALLOWED_ZONES if z != target_zone and z in free_zones]
                    buffer_zone = other_zones[0] if other_zones else "zone_c"
                    
                    plan.append({"skill": "pick", "args": {"object": occupied_by}})
                    plan.append({"skill": "place", "args": {"object": occupied_by, "target": buffer_zone}})

                plan.append({"skill": "pick", "args": {"object": obj}})
                plan.append({"skill": "place", "args": {"object": obj, "target": target_zone}})

            plan.append({"skill": "home", "args": {}})
            return plan

        # 5. Lệnh chỉ pick hoặc chỉ move
        if any(w in cmd_lower for w in ["gắp", "gap", "cầm", "cam", "pick", "lấy", "lay"]) and objs:
            plan.append({"skill": "pick", "args": {"object": objs[0]}})
            plan.append({"skill": "home", "args": {}})
            return plan

        if any(w in cmd_lower for w in ["di chuyển đến", "di chuyen den", "move to", "đi tới", "di toi"]):
            target = zones[0] if zones else (objs[0] if objs else None)
            if target:
                plan.append({"skill": "move_to", "args": {"target": target}})
                return plan

        return []


# ===========================================================================
#  Trình lập kế hoạch chính (LLMPlanner)
# ===========================================================================
class LLMPlanner:
    """Lập kế hoạch đa nền tảng (9Router API, Gemini API, OpenAI API, Offline Heuristic)."""

    def __init__(self, scene_config: dict = None):
        self.scene_cfg = scene_config or {}
        self.offline_planner = OfflineHeuristicPlanner()

        # Tự động nạp file .env nếu có
        self._load_env_file()

        # Kiểm tra API Keys (9Router / OpenAI / Gemini)
        self.openai_key = (
            os.environ.get("LLM_API_KEY") or
            os.environ.get("NINE_ROUTER_API_KEY") or
            os.environ.get("NINEROUTER_API_KEY") or
            os.environ.get("OPENAI_API_KEY") or
            "sk-08ca22f667c9fee6-6b62rx-3eb575a5"
        )
        self.base_url = (
            os.environ.get("LLM_BASE_URL") or
            os.environ.get("NINE_ROUTER_BASE_URL") or
            os.environ.get("ROBOT_LLM_BASE_URL") or
            os.environ.get("OPENAI_BASE_URL") or
            "http://localhost:20128/v1"
        )
        self.model_name = (
            os.environ.get("LLM_MODEL") or
            os.environ.get("ROBOT_LLM_MODEL") or
            "openrouter/nvidia/nemotron-3.5-lightning:free"
        )
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        self.openai_client = None
        if HAS_OPENAI and self.openai_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_key, base_url=self.base_url)
            except Exception:
                self.openai_client = None

    def _load_env_file(self):
        """Đọc và thiết lập các biến từ file .env nếu có."""
        possible_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
            "/home/ductri/workspaces/ur_gz/src/ur3_vlm/.env",
            "/home/ductri/workspaces/ur_gz/src/ur3_llm_control/.env",
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if line.startswith("export "):
                                line = line[7:].strip()
                            if "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                # Expand $ variables if needed
                                if v.startswith("${") and v.endswith("}"):
                                    var_ref = v[2:-1]
                                    v = os.environ.get(var_ref, "")
                                if k not in os.environ and v:
                                    os.environ[k] = v
                except Exception:
                    pass

    def _build_system_prompt(self, world_state: dict = None) -> str:
        """Tạo system prompt kèm trạng thái thế giới từ Perception và few-shot examples."""
        # Trích xuất thông tin WorldState
        state_str = "No active perception state."
        if world_state:
            obj_lines = []
            for obj_name, obj_data in world_state.get("detected_objects", {}).items():
                pos = obj_data.get("position", [0, 0, 0])
                in_z = obj_data.get("in_zone") or "Table"
                obj_lines.append(f"  - {obj_name} ({obj_data.get('color')}): currently at {in_z}, 3D pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

            zone_lines = []
            for zn, zdata in world_state.get("detected_zones", {}).items():
                occ = zdata.get("occupant") or "Empty"
                zone_lines.append(f"  - {zn} ({zdata.get('label')}): Occupant = {occ}")

            state_str = "CURRENT WORLD STATE (from Overhead Camera Perception):\n" + "\n".join(obj_lines) + "\nZones status:\n" + "\n".join(zone_lines)

        return f"""You are a High-Level Task Planner for a 6-DOF UR3e robotic arm operating on a tabletop environment.
Your task is to translate natural language user instructions into a strict, executable JSON array of robot action skills.

=== ENVIRONMENT & OBJECTS ===
Objects (5 dynamic colored cubes): ["red_cube", "yellow_cube", "blue_cube", "green_cube", "purple_cube"]
Zones (4 target placement trays): ["zone_a", "zone_b", "zone_c", "staging_area"] (Note: staging_area is Zone T / Temp buffer)

{state_str}

=== ALLOWED SKILLS WHITELIST (Use ONLY these) ===
1. pick:     {{"skill": "pick", "args": {{"object": "<object_name>"}}}}
2. place:    {{"skill": "place", "args": {{"object": "<object_name>", "target": "<zone_name>"}}}}
3. stack:    {{"skill": "stack", "args": {{"bottom_object": "<object_name>", "top_object": "<object_name>"}}}}
4. unstack:  {{"skill": "unstack", "args": {{"top_object": "<object_name>", "target": "<zone_name>"}}}}
5. move_to:  {{"skill": "move_to", "args": {{"target": "<target_name>"}}}}
6. home:     {{"skill": "home", "args": {{}}}}

=== RULES FOR PLANNING ===
1. Always output ONLY a raw JSON array. No markdown code blocks, no backticks, no explanations.
2. Every pick must be followed by a place or stack. Never place an object without picking it first.
3. If a target zone is already occupied by another block, pick and move the obstructing block to an empty zone first before placing the target block.
4. Always conclude the plan with {{"skill": "home", "args": {{}}}} to return the robot to a safe pose.
5. If the request is invalid, unsafe, or unclear, return an empty array: []

=== FEW-SHOT EXAMPLES ===
Example 1:
User: "Gắp khối màu đỏ đặt vào Zone A"
Assistant: [{{"skill": "pick", "args": {{"object": "red_cube"}}}}, {{"skill": "place", "args": {{"object": "red_cube", "target": "zone_a"}}}}, {{"skill": "home", "args": {{}}}}]

Example 2:
User: "Put the green cube in zone B and blue cube in zone C"
Assistant: [{{"skill": "pick", "args": {{"object": "green_cube"}}}}, {{"skill": "place", "args": {{"object": "green_cube", "target": "zone_b"}}}}, {{"skill": "pick", "args": {{"object": "blue_cube"}}}}, {{"skill": "place", "args": {{"object": "blue_cube", "target": "zone_c"}}}}, {{"skill": "home", "args": {{}}}}]

Example 3:
User: "Đặt khối màu tím lên trên khối màu vàng"
Assistant: [{{"skill": "pick", "args": {{"object": "purple_cube"}}}}, {{"skill": "stack", "args": {{"bottom_object": "yellow_cube", "top_object": "purple_cube"}}}}, {{"skill": "home", "args": {{}}}}]
"""

    def _call_gemini_api(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Gọi Google Gemini REST API."""
        if not self.gemini_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": system_prompt + "\n\nUser Instruction: " + prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
        }
        try:
            resp = requests.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        return None

    def _call_openai_api(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Gọi OpenAI / 9Router API."""
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def validate_and_repair_plan(self, raw_plan: Any, world_state: dict = None) -> Tuple[bool, List[dict], str]:
        """Kiểm tra tính hợp lệ của Action Plan và tự động bổ sung bước an toàn."""
        if not isinstance(raw_plan, list):
            return False, [], "Plan phải là một JSON array"

        repaired_plan = []
        currently_holding = None

        for idx, step in enumerate(raw_plan):
            if not isinstance(step, dict) or "skill" not in step:
                return False, [], f"Bước {idx+1} không có trường 'skill'"

            skill = step["skill"].lower()
            args = step.get("args", {})

            if skill not in ALLOWED_SKILLS:
                return False, [], f"Kỹ năng '{skill}' không nằm trong danh mục cho phép"

            if skill == "pick":
                obj = args.get("object")
                if obj not in ALLOWED_OBJECTS:
                    return False, [], f"Vật thể '{obj}' không hợp lệ"
                currently_holding = obj
                repaired_plan.append({"skill": "pick", "args": {"object": obj}})

            elif skill == "place":
                target = args.get("target") or args.get("zone")
                obj = args.get("object") or currently_holding
                if not obj:
                    return False, [], "Lệnh place nhưng chưa xác định khối đang cầm"
                if target not in ALLOWED_ZONES and target not in ALLOWED_OBJECTS:
                    return False, [], f"Vị trí đích '{target}' không hợp lệ"
                repaired_plan.append({"skill": "place", "args": {"object": obj, "target": target}})
                currently_holding = None

            elif skill == "stack":
                bot = args.get("bottom_object")
                top = args.get("top_object") or currently_holding
                repaired_plan.append({"skill": "stack", "args": {"bottom_object": bot, "top_object": top}})
                currently_holding = None

            elif skill in ["unstack", "move_to", "home"]:
                repaired_plan.append(step)

        # Tự động thêm 'home' ở cuối nếu thiếu
        if repaired_plan and repaired_plan[-1]["skill"] != "home":
            repaired_plan.append({"skill": "home", "args": {}})

        return True, repaired_plan, "Plan hợp lệ"

    def plan(self, command: str, world_state: dict = None) -> List[dict]:
        """
        Lập kế hoạch từ câu lệnh ngôn ngữ tự nhiên.
        Tự động fallback về Offline Heuristic nếu không có API.
        """
        system_prompt = self._build_system_prompt(world_state)
        raw_text = None

        # 1. Thử Gemini API
        if self.gemini_key:
            raw_text = self._call_gemini_api(command, system_prompt)

        # 2. Thử OpenAI / 9Router API nếu chưa có kết quả
        if not raw_text and self.openai_client:
            raw_text = self._call_openai_api(command, system_prompt)

        # 3. Phân tích kết quả JSON từ LLM
        if raw_text:
            try:
                # Xóa markdown formatting nếu LLM trả về ```json ... ```
                clean_json = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                parsed = json.loads(clean_json)
                is_valid, final_plan, msg = self.validate_and_repair_plan(parsed, world_state)
                if is_valid and len(final_plan) > 0:
                    return final_plan
            except Exception:
                pass

        # 4. Fallback về Bộ lập kế hoạch cục bộ Offline Heuristic
        heuristic_plan = self.offline_planner.plan(command, world_state)
        is_valid, final_plan, _ = self.validate_and_repair_plan(heuristic_plan, world_state)
        return final_plan if is_valid else []


# ===========================================================================
#  CLI Test Interactive
# ===========================================================================
def main(args=None):
    planner = LLMPlanner()
    print("=" * 60)
    print("  UR3e VLM Task Planner (Phase 3 — Interactive / Test)")
    print("=" * 60)
    
    # Mock world state for testing
    mock_state = {
        "detected_objects": {
            "red_cube": {"color": "red", "in_zone": None, "position": [0.30, -0.20, 0.02]},
            "yellow_cube": {"color": "yellow", "in_zone": "zone_a", "position": [0.45, 0.20, 0.02]},
            "blue_cube": {"color": "blue", "in_zone": None, "position": [0.30, 0.20, 0.02]},
            "green_cube": {"color": "green", "in_zone": None, "position": [0.24, -0.10, 0.02]},
            "purple_cube": {"color": "purple", "in_zone": None, "position": [0.24, 0.10, 0.02]},
        },
        "detected_zones": {
            "zone_a": {"label": "Zone A", "occupant": "yellow_cube"},
            "zone_b": {"label": "Zone B", "occupant": None},
            "zone_c": {"label": "Zone C", "occupant": None},
        },
        "summary": {"occupied_zones": ["zone_a"], "free_zones": ["zone_b", "zone_c"]},
    }

    test_cmds = [
        "Gắp khối màu đỏ đặt vào Zone B",
        "Put the green cube in zone C",
        "Đặt khối màu tím lên trên khối màu xanh dương",
        "Đưa khối màu vàng vào Zone B",
        "Về vị trí an toàn",
    ]

    print("\n--- Chạy thử nghiệm các câu lệnh mẫu ---")
    for cmd in test_cmds:
        print(f"\n[User]: {cmd}")
        plan_res = planner.plan(cmd, mock_state)
        print(f"[Plan]: {json.dumps(plan_res, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 60)
    print("Phase 3 - LLM/VLM Task Planner đã sẵn sàng!")
    print("=" * 60)


if __name__ == "__main__":
    main()
