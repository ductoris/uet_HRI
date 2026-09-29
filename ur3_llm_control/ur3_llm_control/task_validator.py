#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_validator.py — Kiểm tra tính hợp lệ của plan JSON do LLM sinh ra.

Chức năng:
  1. Parse & validate cú pháp JSON.
  2. Đối chiếu whitelist: skill names, object names, zone names.
  3. Kiểm tra logic tuần tự:
     - Không place trước pick.
     - Không pick khi đang giữ vật.
     - place đúng vật đang giữ.
     - Kết thúc bằng home().
  4. Kiểm tra số bước không vượt MAX_PLAN_STEPS.
  5. Mô phỏng world state để phát hiện xung đột zone.
  6. Trả về danh sách lỗi chi tiết hoặc xác nhận plan hợp lệ.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import json
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Whitelist
# ---------------------------------------------------------------------------
ALLOWED_SKILLS = {"pick", "place", "home", "move_above", "open_gripper", "close_gripper"}
ALLOWED_OBJECTS = {"red_cube", "yellow_cube", "blue_cube"}
ALLOWED_ZONES = {"zone_a", "zone_b", "zone_c", "temp_zone"}

MAX_PLAN_STEPS = 30   # Giới hạn số bước tối đa trong một plan


# ---------------------------------------------------------------------------
#  Lớp ValidationError
# ---------------------------------------------------------------------------
class ValidationError:
    """Mô tả một lỗi validation."""

    def __init__(self, step_index: int, message: str):
        self.step_index = step_index
        self.message = message

    def __repr__(self):
        return f"Step {self.step_index}: {self.message}"

    def __str__(self):
        return self.__repr__()


# ---------------------------------------------------------------------------
#  Hàm parse JSON
# ---------------------------------------------------------------------------
def parse_llm_response(raw_text: str) -> Tuple[Optional[list], Optional[str]]:
    """
    Parse phản hồi LLM thành danh sách skill steps.

    Xử lý:
      - Markdown code block (```json ... ```)
      - List trực tiếp: [{"skill": ...}, ...]
      - Dict wrapper: {"plan": [...]} hoặc {"steps": [...]}
      - Chuỗi rỗng / unrelated → trả về list rỗng (plan rỗng hợp lệ)

    Parameters
    ----------
    raw_text : str
        Phản hồi thô từ LLM.

    Returns
    -------
    (list | None, str | None)
        - (parsed_list, None) nếu thành công (kể cả list rỗng).
        - (None, error_message) nếu parse thất bại.
    """
    text = raw_text.strip()

    if not text:
        return [], None  # Plan rỗng = hợp lệ (không thực hiện gì)

    # Loại bỏ markdown code block nếu có
    if text.startswith("```"):
        lines = text.split("\n")
        filtered = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not stripped.startswith("```"):
                filtered.append(line)
        text = "\n".join(filtered).strip()

    # Xóa ``` còn sót lại
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    # Hỗ trợ dict wrapper {"plan": [...]} và {"steps": [...]}
    if isinstance(parsed, dict):
        if "steps" in parsed:
            parsed = parsed["steps"]
        elif "plan" in parsed:
            parsed = parsed["plan"]
        else:
            return None, "JSON object không chứa key 'steps' hoặc 'plan'."

    if not isinstance(parsed, list):
        return None, "Kết quả JSON không phải là một danh sách (list)."

    return parsed, None


# ---------------------------------------------------------------------------
#  Validate plan
# ---------------------------------------------------------------------------
def validate_plan(
    steps: list,
    initial_world_state: Dict[str, str] = None,
) -> Tuple[bool, List[ValidationError]]:
    """
    Kiểm tra tính hợp lệ toàn diện của plan.

    Parameters
    ----------
    steps : list
        Danh sách dict, mỗi dict có key 'skill' và tùy chọn 'args'.
        Ví dụ:
        [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
            {"skill": "home", "args": {}}
        ]
    initial_world_state : dict | None
        Trạng thái ban đầu: {object_name: "table" | zone_name | "gripper"}.
        Dùng để phát hiện xung đột zone. Nếu None, bỏ qua kiểm tra này.

    Returns
    -------
    (bool, List[ValidationError])
        - (True, []) nếu plan hợp lệ.
        - (False, [errors]) nếu có lỗi.
    """
    errors: List[ValidationError] = []
    holding: Optional[str] = None          # Vật đang giữ (mô phỏng)

    # World state mô phỏng: {object: location}
    # location = "table" | "zone_a" | ... | "gripper"
    world: Dict[str, str] = {}
    if initial_world_state:
        world = dict(initial_world_state)
    else:
        for obj in ALLOWED_OBJECTS:
            world.setdefault(obj, "table")

    # Zone occupancy: {zone_name: object_name | None}
    zone_occupied: Dict[str, Optional[str]] = {z: None for z in ALLOWED_ZONES}
    for obj, loc in world.items():
        if loc in ALLOWED_ZONES:
            zone_occupied[loc] = obj

    # --- Plan rỗng: hợp lệ (không thực thi gì) ---
    if len(steps) == 0:
        return True, []

    # --- Giới hạn số bước ---
    if len(steps) > MAX_PLAN_STEPS:
        errors.append(ValidationError(
            0,
            f"Plan có {len(steps)} bước, vượt quá giới hạn tối đa {MAX_PLAN_STEPS}."
        ))

    # --- Validate từng bước ---
    for idx, step in enumerate(steps):
        step_num = idx + 1

        if not isinstance(step, dict):
            errors.append(ValidationError(step_num, f"Step không phải dict: {step}"))
            continue

        skill = step.get("skill")
        args = step.get("args", {})

        if skill is None:
            errors.append(ValidationError(step_num, "Thiếu key 'skill'."))
            continue

        # --- Kiểm tra skill name ---
        if skill not in ALLOWED_SKILLS:
            errors.append(ValidationError(
                step_num,
                f"Skill '{skill}' KHÔNG hợp lệ. Whitelist: {sorted(ALLOWED_SKILLS)}"
            ))
            continue  # Không kiểm tra args của skill không hợp lệ

        # --- Kiểm tra args theo từng skill ---
        if skill == "pick":
            obj = args.get("object")
            if obj is None:
                errors.append(ValidationError(step_num, "pick() thiếu tham số 'object'."))
            elif obj not in ALLOWED_OBJECTS:
                errors.append(ValidationError(
                    step_num,
                    f"pick(): object '{obj}' KHÔNG hợp lệ. "
                    f"Cho phép: {sorted(ALLOWED_OBJECTS)}"
                ))
            else:
                # Logic: không pick khi đang giữ vật khác
                if holding is not None:
                    errors.append(ValidationError(
                        step_num,
                        f"pick({obj}): Đang giữ '{holding}' — "
                        f"phải place trước khi pick tiếp."
                    ))
                else:
                    holding = obj
                    # Xóa vật khỏi zone hiện tại khi pick
                    old_loc = world.get(obj, "table")
                    if old_loc in zone_occupied and zone_occupied.get(old_loc) == obj:
                        zone_occupied[old_loc] = None
                    world[obj] = "gripper"

        elif skill == "place":
            obj = args.get("object")
            zone = args.get("zone")

            obj_valid = True
            zone_valid = True

            if obj is None:
                errors.append(ValidationError(step_num, "place() thiếu tham số 'object'."))
                obj_valid = False
            elif obj not in ALLOWED_OBJECTS:
                errors.append(ValidationError(
                    step_num,
                    f"place(): object '{obj}' KHÔNG hợp lệ. "
                    f"Cho phép: {sorted(ALLOWED_OBJECTS)}"
                ))
                obj_valid = False

            if zone is None:
                errors.append(ValidationError(step_num, "place() thiếu tham số 'zone'."))
                zone_valid = False
            elif zone not in ALLOWED_ZONES:
                errors.append(ValidationError(
                    step_num,
                    f"place(): zone '{zone}' KHÔNG hợp lệ. "
                    f"Cho phép: {sorted(ALLOWED_ZONES)}"
                ))
                zone_valid = False

            if obj_valid and zone_valid:
                # Logic: phải đang giữ đúng vật
                if holding is None:
                    errors.append(ValidationError(
                        step_num,
                        f"place({obj}, {zone}): Chưa pick vật nào — "
                        f"phải pick trước khi place."
                    ))
                elif holding != obj:
                    errors.append(ValidationError(
                        step_num,
                        f"place({obj}, {zone}): Đang giữ '{holding}', "
                        f"không phải '{obj}'."
                    ))
                else:
                    # Kiểm tra zone xung đột (zone đã bị vật khác chiếm)
                    occupant = zone_occupied.get(zone)
                    if occupant is not None and occupant != obj:
                        errors.append(ValidationError(
                            step_num,
                            f"place({obj}, {zone}): Zone '{zone}' đang bị '{occupant}' chiếm. "
                            f"Cần di chuyển '{occupant}' trước, hoặc dùng temp_zone."
                        ))
                    else:
                        # Cập nhật world state mô phỏng
                        # Xóa vật khỏi zone cũ nếu có
                        old_loc = world.get(obj)
                        if old_loc and old_loc in zone_occupied:
                            if zone_occupied[old_loc] == obj:
                                zone_occupied[old_loc] = None
                        # Đặt vật vào zone mới
                        zone_occupied[zone] = obj
                        world[obj] = zone
                        holding = None

        elif skill == "move_above":
            target = (
                args.get("target")
                or args.get("object")
                or args.get("zone")
            )
            if target is None:
                errors.append(ValidationError(
                    step_num,
                    "move_above() thiếu 'target' (hoặc 'object'/'zone')."
                ))
            elif target not in ALLOWED_OBJECTS and target not in ALLOWED_ZONES:
                errors.append(ValidationError(
                    step_num,
                    f"move_above(): target '{target}' KHÔNG hợp lệ. "
                    f"Cho phép: objects + zones."
                ))

        elif skill in ("home", "open_gripper", "close_gripper"):
            pass  # Không cần args đặc biệt

    # --- Kiểm tra kết thúc bằng home() ---
    if len(steps) > 0 and len(errors) == 0:
        last_skill = steps[-1].get("skill") if isinstance(steps[-1], dict) else None
        if last_skill != "home":
            errors.append(ValidationError(
                len(steps),
                f"Plan phải kết thúc bằng 'home()'. Skill cuối: '{last_skill}'."
            ))

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
#  Tiện ích: format plan thành text readable
# ---------------------------------------------------------------------------
def format_plan_text(steps: list) -> str:
    """
    Chuyển đổi plan JSON thành dạng text dễ đọc cho terminal.

    Ví dụ output:
        pick(red_cube)
        place(red_cube, zone_b)
        home()
    """
    if not steps:
        return "(plan rỗng — không thực thi)"

    lines = []
    for step in steps:
        skill = step.get("skill", "?")
        args = step.get("args", {})

        if skill == "pick":
            obj = args.get("object", "?")
            lines.append(f"pick({obj})")
        elif skill == "place":
            obj = args.get("object", "?")
            zone = args.get("zone", "?")
            lines.append(f"place({obj}, {zone})")
        elif skill == "move_above":
            target = args.get("target") or args.get("object") or args.get("zone", "?")
            lines.append(f"move_above({target})")
        elif skill in ("home", "open_gripper", "close_gripper"):
            lines.append(f"{skill}()")
        else:
            lines.append(f"{skill}({args})")

    return "\n".join(lines)
