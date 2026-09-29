# ur3_llm_control/__init__.py
"""
ur3_llm_control — Điều khiển robot UR3/UR3e bằng LLM và Skill-based Planning.

Sinh viên: Mai Duc Tri | MSSV: 23020776 | P = 4
  Zone A = blue_cube, Zone B = red_cube, Zone C = yellow_cube

Kiến trúc:
  llm_planner    → Gọi 9Router API, sinh chuỗi skill JSON
  task_validator  → Kiểm tra tính hợp lệ cú pháp & logic
  robot_skills    → Thực thi chuyển động qua MoveIt 2
  skill_executor  → Node chính, điều phối toàn bộ luồng
"""
