#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_task_validator.py — Unit tests cho TaskValidator.

Bao gồm:
  - Validator: valid/invalid skills, objects, zones
  - Validator: logic tuần tự (pick/place ordering)
  - Validator: kết thúc bằng home()
  - Validator: giới hạn số bước
  - Validator: xung đột zone
  - parse_llm_response: các dạng JSON input

Chạy:
  cd ~/workspaces/ur_gz
  source install/setup.bash
  cd src/ur3_llm_control
  python -m pytest test/test_task_validator.py -v

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import sys
import os

# Thêm package vào path để import không cần ROS 2 build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ur3_llm_control.task_validator import (
    parse_llm_response,
    validate_plan,
    format_plan_text,
    ALLOWED_SKILLS,
    ALLOWED_OBJECTS,
    ALLOWED_ZONES,
    MAX_PLAN_STEPS,
)


# ============================================================
#  Fixtures: Plan mẫu
# ============================================================
VALID_PLAN_SIMPLE = [
    {"skill": "pick", "args": {"object": "red_cube"}},
    {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
    {"skill": "home", "args": {}},
]

VALID_PLAN_ARRANGE = [
    {"skill": "pick", "args": {"object": "blue_cube"}},
    {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},
    {"skill": "pick", "args": {"object": "red_cube"}},
    {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
    {"skill": "pick", "args": {"object": "yellow_cube"}},
    {"skill": "place", "args": {"object": "yellow_cube", "zone": "zone_c"}},
    {"skill": "home", "args": {}},
]

VALID_PLAN_WITH_TEMP = [
    {"skill": "pick", "args": {"object": "red_cube"}},
    {"skill": "place", "args": {"object": "red_cube", "zone": "temp_zone"}},
    {"skill": "pick", "args": {"object": "blue_cube"}},
    {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},
    {"skill": "pick", "args": {"object": "red_cube"}},
    {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
    {"skill": "home", "args": {}},
]

VALID_PLAN_HOME_ONLY = [
    {"skill": "home", "args": {}},
]

VALID_PLAN_EMPTY = []  # Plan rỗng hợp lệ


# ============================================================
#  Tests: parse_llm_response
# ============================================================
class TestParseLLMResponse:
    def test_parse_valid_array(self):
        raw = '[{"skill": "home", "args": {}}]'
        steps, err = parse_llm_response(raw)
        assert steps is not None
        assert err is None
        assert len(steps) == 1
        assert steps[0]["skill"] == "home"

    def test_parse_markdown_code_block(self):
        raw = '```json\n[{"skill": "home", "args": {}}]\n```'
        steps, err = parse_llm_response(raw)
        assert steps is not None
        assert err is None

    def test_parse_plan_wrapper(self):
        raw = '{"plan": [{"skill": "home", "args": {}}]}'
        steps, err = parse_llm_response(raw)
        assert steps is not None
        assert len(steps) == 1

    def test_parse_steps_wrapper(self):
        raw = '{"steps": [{"skill": "home", "args": {}}]}'
        steps, err = parse_llm_response(raw)
        assert steps is not None
        assert len(steps) == 1

    def test_parse_empty_array(self):
        raw = "[]"
        steps, err = parse_llm_response(raw)
        assert steps == []
        assert err is None

    def test_parse_empty_string(self):
        raw = ""
        steps, err = parse_llm_response(raw)
        assert steps == []
        assert err is None

    def test_parse_invalid_json(self):
        raw = "this is not json"
        steps, err = parse_llm_response(raw)
        assert steps is None
        assert "JSON parse error" in err

    def test_parse_invalid_dict_no_key(self):
        raw = '{"foo": "bar"}'
        steps, err = parse_llm_response(raw)
        assert steps is None

    def test_parse_plain_string_result(self):
        raw = '"just a string"'
        steps, err = parse_llm_response(raw)
        assert steps is None


# ============================================================
#  Tests: validate_plan — Valid plans
# ============================================================
class TestValidatePlanValid:
    def test_simple_pick_place_home(self):
        ok, errors = validate_plan(VALID_PLAN_SIMPLE)
        assert ok, f"Expected valid but got errors: {errors}"
        assert errors == []

    def test_arrange_three_objects(self):
        ok, errors = validate_plan(VALID_PLAN_ARRANGE)
        assert ok, f"Expected valid: {errors}"

    def test_with_temp_zone(self):
        ok, errors = validate_plan(VALID_PLAN_WITH_TEMP)
        assert ok, f"Expected valid: {errors}"

    def test_home_only(self):
        ok, errors = validate_plan(VALID_PLAN_HOME_ONLY)
        assert ok, f"Expected valid: {errors}"

    def test_empty_plan(self):
        ok, errors = validate_plan(VALID_PLAN_EMPTY)
        assert ok, f"Empty plan should be valid: {errors}"
        assert errors == []

    def test_move_above_valid_object(self):
        plan = [
            {"skill": "move_above", "args": {"target": "red_cube"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert ok, f"Expected valid: {errors}"

    def test_move_above_valid_zone(self):
        plan = [
            {"skill": "move_above", "args": {"target": "zone_a"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert ok, f"Expected valid: {errors}"

    def test_all_three_zones_arranged(self):
        plan = [
            {"skill": "pick", "args": {"object": "blue_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
            {"skill": "pick", "args": {"object": "yellow_cube"}},
            {"skill": "place", "args": {"object": "yellow_cube", "zone": "zone_c"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert ok, f"Arrange plan should be valid: {errors}"


# ============================================================
#  Tests: validate_plan — Invalid skill names
# ============================================================
class TestValidatePlanInvalidSkill:
    def test_unknown_skill_fly(self):
        plan = [
            {"skill": "fly", "args": {}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("fly" in str(e) for e in errors)

    def test_unknown_skill_teleport(self):
        plan = [
            {"skill": "teleport", "args": {"object": "red_cube"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("teleport" in str(e) for e in errors)

    def test_unknown_skill_throw(self):
        plan = [
            {"skill": "throw", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok

    def test_case_sensitive_skill(self):
        """Skill names phải lowercase chính xác."""
        plan = [
            {"skill": "Pick", "args": {"object": "red_cube"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok


# ============================================================
#  Tests: validate_plan — Invalid objects
# ============================================================
class TestValidatePlanInvalidObject:
    def test_invalid_object_green_cube(self):
        plan = [
            {"skill": "pick", "args": {"object": "green_cube"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("green_cube" in str(e) for e in errors)

    def test_invalid_object_white_cube(self):
        plan = [
            {"skill": "pick", "args": {"object": "white_cube"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok

    def test_missing_object_key(self):
        plan = [
            {"skill": "pick", "args": {}},  # Thiếu 'object'
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("thiếu" in str(e).lower() or "object" in str(e).lower() for e in errors)


# ============================================================
#  Tests: validate_plan — Invalid zones
# ============================================================
class TestValidatePlanInvalidZone:
    def test_invalid_zone_d(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_d"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("zone_d" in str(e) for e in errors)

    def test_invalid_zone_x(self):
        plan = [
            {"skill": "pick", "args": {"object": "blue_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_x"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok

    def test_missing_zone_key(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube"}},  # Thiếu 'zone'
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok


# ============================================================
#  Tests: validate_plan — Logic ordering
# ============================================================
class TestValidatePlanLogic:
    def test_place_without_pick(self):
        plan = [
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_a"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("pick" in str(e).lower() for e in errors)

    def test_pick_while_holding(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "pick", "args": {"object": "blue_cube"}},  # Sai
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("giữ" in str(e) or "holding" in str(e).lower() for e in errors)

    def test_place_wrong_object(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},  # Sai
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok

    def test_must_end_with_home(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
            # Thiếu home()
        ]
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("home" in str(e).lower() for e in errors)

    def test_end_with_pick_not_home(self):
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
        ]
        ok, errors = validate_plan(plan)
        assert not ok

    def test_max_steps_exceeded(self):
        plan = [{"skill": "home", "args": {}}] * (MAX_PLAN_STEPS + 1)
        ok, errors = validate_plan(plan)
        assert not ok
        assert any("giới hạn" in str(e).lower() or "max" in str(e).lower() for e in errors)


# ============================================================
#  Tests: Zone conflict via world_state
# ============================================================
class TestZoneConflict:
    def test_zone_conflict_detected(self):
        """place blue_cube vào zone_a khi zone_a đã có red_cube → lỗi."""
        initial_state = {
            "red_cube": "zone_a",
            "yellow_cube": "table",
            "blue_cube": "table",
        }
        plan = [
            {"skill": "pick", "args": {"object": "blue_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},  # Conflict!
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan, initial_world_state=initial_state)
        assert not ok
        assert any("chiếm" in str(e) or "zone_a" in str(e) for e in errors)

    def test_no_conflict_when_zone_empty(self):
        """place vào zone trống → không lỗi."""
        initial_state = {
            "red_cube": "table",
            "yellow_cube": "table",
            "blue_cube": "table",
        }
        plan = [
            {"skill": "pick", "args": {"object": "blue_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan, initial_world_state=initial_state)
        assert ok, f"No conflict expected: {errors}"

    def test_conflict_resolved_with_temp_zone(self):
        """Dùng temp_zone để giải xung đột → plan hợp lệ."""
        # red_cube ở zone_a, ta muốn đặt blue_cube vào zone_a
        # → cần chuyển red_cube sang temp_zone trước
        initial_state = {
            "red_cube": "zone_a",
            "yellow_cube": "table",
            "blue_cube": "table",
        }
        plan = [
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "temp_zone"}},
            {"skill": "pick", "args": {"object": "blue_cube"}},
            {"skill": "place", "args": {"object": "blue_cube", "zone": "zone_a"}},
            {"skill": "pick", "args": {"object": "red_cube"}},
            {"skill": "place", "args": {"object": "red_cube", "zone": "zone_b"}},
            {"skill": "home", "args": {}},
        ]
        ok, errors = validate_plan(plan, initial_world_state=initial_state)
        assert ok, f"Temp zone conflict resolution should be valid: {errors}"


# ============================================================
#  Tests: format_plan_text
# ============================================================
class TestFormatPlanText:
    def test_format_simple(self):
        text = format_plan_text(VALID_PLAN_SIMPLE)
        assert "pick(red_cube)" in text
        assert "place(red_cube, zone_b)" in text
        assert "home()" in text

    def test_format_empty(self):
        text = format_plan_text([])
        assert "rỗng" in text or "empty" in text.lower() or "plan" in text.lower()

    def test_format_move_above(self):
        plan = [
            {"skill": "move_above", "args": {"target": "zone_a"}},
            {"skill": "home", "args": {}},
        ]
        text = format_plan_text(plan)
        assert "move_above(zone_a)" in text

    def test_format_gripper(self):
        plan = [
            {"skill": "open_gripper", "args": {}},
            {"skill": "close_gripper", "args": {}},
            {"skill": "home", "args": {}},
        ]
        text = format_plan_text(plan)
        assert "open_gripper()" in text
        assert "close_gripper()" in text


# ============================================================
#  Tests: Mock LLM / Fallback (standalone, không cần ROS 2)
# ============================================================
# Tách fallback logic ra để test mà không cần ament_index_python
def _mock_fallback_plan(user_command: str, student_cfg: dict) -> list:
    """
    Phiên bản standalone của _fallback_plan để test mà không cần ROS 2.
    Logic y chang llm_planner._fallback_plan nhưng nhận student_cfg trực tiếp.
    """
    cmd_lower = user_command.lower().strip()
    target = student_cfg.get("target_assignment", {})

    # 1. Yêu cầu sắp xếp theo MSSV
    student_triggers = [
        "student", "mssv", "đề bài", "de bai", "arrange", "sắp xếp", "sap xep",
        "theo mssv", "theo đề", "23020776"
    ]
    if any(k in cmd_lower for k in student_triggers):
        plan = []
        for zone_key in ["zone_a", "zone_b", "zone_c"]:
            obj = target.get(zone_key, "blue_cube")
            plan.append({"skill": "pick", "args": {"object": obj}})
            plan.append({"skill": "place", "args": {"object": obj, "zone": zone_key}})
        plan.append({"skill": "home", "args": {}})
        return plan

    # 2. Home
    if any(k in cmd_lower for k in ["home", "về vị trí", "ve vi tri"]):
        return [{"skill": "home", "args": {}}]

    # 3. Nhận diện đối tượng
    obj = None
    if "blue" in cmd_lower or "xanh dương" in cmd_lower or "xanh" in cmd_lower:
        obj = "blue_cube"
    elif "red" in cmd_lower or "đỏ" in cmd_lower or "do" in cmd_lower:
        obj = "red_cube"
    elif "yellow" in cmd_lower or "vàng" in cmd_lower or "vang" in cmd_lower:
        obj = "yellow_cube"

    # 4. Nhận diện khu vực
    zone = None
    if any(k in cmd_lower for k in ["zone a", "zone_a", "khay a", "ô a", "o a", "vào a", " a"]):
        zone = "zone_a"
    elif any(k in cmd_lower for k in ["zone b", "zone_b", "khay b", "ô b", "o b", "vào b", " b"]):
        zone = "zone_b"
    elif any(k in cmd_lower for k in ["zone c", "zone_c", "khay c", "ô c", "o c", "vào c", " c"]):
        zone = "zone_c"

    # 5. Ghép thành plan
    if obj and zone:
        return [
            {"skill": "pick", "args": {"object": obj}},
            {"skill": "place", "args": {"object": obj, "zone": zone}},
            {"skill": "home", "args": {}},
        ]

    return None  # Không nhận diện được


MOCK_STUDENT_CFG = {
    "student": {"name": "Test", "mssv": "23020776", "P": 4, "last_two_digits": 76},
    "target_assignment": {
        "zone_a": "blue_cube",
        "zone_b": "red_cube",
        "zone_c": "yellow_cube",
    },
}


class TestMockLLMPlanner:
    """
    Test Fallback Planner mà không cần ROS 2 node hay API key.
    Kiểm tra logic nhận diện lệnh VI/EN đa dạng (standalone).
    """

    def test_fallback_english_red_to_zone_b(self):
        steps = _mock_fallback_plan("Put the red cube in zone B.", MOCK_STUDENT_CFG)
        assert steps is not None
        skills = [s["skill"] for s in steps]
        assert "pick" in skills
        assert "place" in skills
        obj_picked = next(s["args"]["object"] for s in steps if s["skill"] == "pick")
        assert obj_picked == "red_cube"

    def test_fallback_english_blue_to_zone_a(self):
        steps = _mock_fallback_plan("Move the blue cube to zone A.", MOCK_STUDENT_CFG)
        assert steps is not None
        obj_picked = next((s["args"]["object"] for s in steps if s["skill"] == "pick"), None)
        assert obj_picked == "blue_cube"

    def test_fallback_vietnamese_red(self):
        steps = _mock_fallback_plan("Đưa khối màu đỏ vào vùng B.", MOCK_STUDENT_CFG)
        assert steps is not None
        obj_picked = next((s["args"]["object"] for s in steps if s["skill"] == "pick"), None)
        assert obj_picked == "red_cube"

    def test_fallback_vietnamese_yellow_zone_a(self):
        steps = _mock_fallback_plan("Hãy lấy khối màu vàng và đặt nó vào ô A.", MOCK_STUDENT_CFG)
        assert steps is not None
        obj_picked = next((s["args"]["object"] for s in steps if s["skill"] == "pick"), None)
        assert obj_picked == "yellow_cube"

    def test_fallback_student_id_arrange(self):
        steps = _mock_fallback_plan(
            "Arrange all objects according to my student ID.", MOCK_STUDENT_CFG
        )
        assert steps is not None
        assert len(steps) > 0
        picks = [s for s in steps if s["skill"] == "pick"]
        assert len(picks) == 3, f"Expected 3 picks, got {len(picks)}"
        # Kiểm tra đủ 3 vật
        picked_objs = {s["args"]["object"] for s in picks}
        assert picked_objs == {"red_cube", "yellow_cube", "blue_cube"}

    def test_fallback_home_command(self):
        steps = _mock_fallback_plan("Go home.", MOCK_STUDENT_CFG)
        assert steps is not None
        assert steps[0]["skill"] == "home"

    def test_fallback_mssv_trigger(self):
        steps = _mock_fallback_plan("Sắp xếp theo MSSV.", MOCK_STUDENT_CFG)
        assert steps is not None
        picks = [s for s in steps if s["skill"] == "pick"]
        assert len(picks) == 3

    def test_fallback_unrelated_returns_none(self):
        steps = _mock_fallback_plan("What is the weather today?", MOCK_STUDENT_CFG)
        assert steps is None  # Fallback không nhận diện được → None

    def test_fallback_plan_valid_after_generation(self):
        """Plan từ fallback phải pass validate_plan."""
        steps = _mock_fallback_plan("Put the red cube in zone B.", MOCK_STUDENT_CFG)
        assert steps is not None
        ok, errors = validate_plan(steps)
        assert ok, f"Fallback plan should be valid: {errors}"

    def test_fallback_arrange_plan_valid(self):
        """Plan sắp xếp theo MSSV phải pass validate_plan."""
        steps = _mock_fallback_plan(
            "Arrange all objects according to my student ID.", MOCK_STUDENT_CFG
        )
        assert steps is not None
        ok, errors = validate_plan(steps)
        assert ok, f"Arrange plan should be valid: {errors}"


# ============================================================
#  Tests: Conflict Resolution Algorithm
# ============================================================
class TestConflictResolution:
    def test_single_target_with_occupant(self):
        """Khi target zone bị chiếm, thuật toán phải dùng temp_zone."""
        from ur3_llm_control.llm_planner import resolve_conflicts_and_generate_plan
        targets = {"blue_cube": "zone_a"}
        world_state = {"yellow_cube": "zone_a", "blue_cube": "table", "red_cube": "table"}

        steps = resolve_conflicts_and_generate_plan(targets, current_world_state=world_state)
        ok, errors = validate_plan(steps, initial_world_state=world_state)
        assert ok, f"Resolved plan must be valid: {errors}"

        # Kiểm tra có dùng temp_zone cho yellow_cube
        temp_places = [s for s in steps if s.get("skill") == "place" and s["args"].get("zone") == "temp_zone"]
        assert len(temp_places) == 1
        assert temp_places[0]["args"]["object"] == "yellow_cube"

    def test_arrange_all_with_occupied_zone(self):
        """Khi sắp xếp 3 khối và có khối bị trùng zone, plan tự gỡ xung đột."""
        from ur3_llm_control.llm_planner import resolve_conflicts_and_generate_plan
        targets = {"blue_cube": "zone_a", "red_cube": "zone_b", "yellow_cube": "zone_c"}
        world_state = {"yellow_cube": "zone_a", "blue_cube": "table", "red_cube": "table"}

        steps = resolve_conflicts_and_generate_plan(targets, current_world_state=world_state)
        ok, errors = validate_plan(steps, initial_world_state=world_state)
        assert ok, f"Resolved plan for arrange must be valid: {errors}"

    def test_cycle_conflict_resolution(self):
        """Khi có chu kỳ xung đột (red ở zone_a, blue ở zone_b), thuật toán giải quyết bằng temp_zone."""
        from ur3_llm_control.llm_planner import resolve_conflicts_and_generate_plan
        targets = {"blue_cube": "zone_a", "red_cube": "zone_b", "yellow_cube": "zone_c"}
        world_state = {"red_cube": "zone_a", "blue_cube": "zone_b", "yellow_cube": "table"}

        steps = resolve_conflicts_and_generate_plan(targets, current_world_state=world_state)
        ok, errors = validate_plan(steps, initial_world_state=world_state)
        assert ok, f"Cycle conflict resolution plan must be valid: {errors}"


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

