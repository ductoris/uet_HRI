#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ur3_kinematics.py — Forward & Inverse Kinematics cho robot UR3/UR3e (Bài 03).

Cung cấp:
  1. fk(q): Tính Pose (vị trí + ma trận quay 4x4) của tool0 từ 6 giá trị góc khớp.
  2. solve_ik(target_xyz, seed=None): Giải Inverse Kinematics tìm 6 góc khớp
     sao cho tool0 tại target_xyz và đầu công tác chúc thẳng xuống (-Z).
     Đảm bảo nghiệm thuộc giới hạn vật lý của UR3e, liên tục với seed để
     triệt tiêu hoàn toàn hiện tượng vung tay / lật khuỷu / lật vai bất thường.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
#  Thông số hình học UR3e (theo chuẩn Universal Robots & URDF ROS 2)
# ---------------------------------------------------------------------------
D1 = 0.15185       # Base to shoulder
A2 = -0.24355      # Upper arm length
A3 = -0.2132       # Forearm length
D4 = 0.13105       # Wrist 1 offset
D5 = 0.08535       # Wrist 2 offset
D6 = 0.0921        # Wrist 3 to tool0

# Home configuration chuẩn
HOME_JOINTS = [0.0, -1.570796, 1.570796, -1.570796, -1.570796, 0.0]

# Giới hạn khớp an toàn cho UR3e (Elbow-Up, đầu kẹp chúc thẳng xuống, không bị lật khuỷu/lật vai)
UR3E_BOUNDS = [
    (-np.pi, np.pi),         # shoulder_pan (hướng về phía bàn X > 0)
    (-np.pi + 0.15, -0.15),  # shoulder_lift (LUÔN ÂM, vươn về phía trước/trên)
    (0.15, np.pi - 0.15),    # elbow (LUÔN DƯƠNG -> Elbow-Up)
    (-np.pi, 0.0),           # wrist_1 (uốn xuống)
    (-1.65, -1.50),          # wrist_2 (khóa cố định quanh -pi/2 để kẹp thẳng đứng)
    (-np.pi, np.pi),         # wrist_3
]


def _rot_z(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0.0, 0.0], [sy, cy, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp, 0.0], [0.0, 1.0, 0.0, 0.0], [-sp, 0.0, cp, 0.0], [0.0, 0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cr, -sr, 0.0], [0.0, sr, cr, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def fk(q: List[float]) -> np.ndarray:
    """
    Tính ma trận biến đổi 4x4 từ base_link tới tool0.
    Khớp base_link -> base_link_inertia trong URDF xoay pi rad quanh trục Z.
    """
    T = _rot_z(np.pi)
    # base to shoulder
    T = T @ _trans(0.0, 0.0, D1) @ _rot_z(q[0])
    # shoulder to upper_arm
    T = T @ _trans(0.0, 0.0, 0.0) @ _rpy(np.pi / 2.0, 0.0, 0.0) @ _rot_z(q[1])
    # upper_arm to forearm
    T = T @ _trans(A2, 0.0, 0.0) @ _rpy(0.0, 0.0, 0.0) @ _rot_z(q[2])
    # forearm to wrist_1
    T = T @ _trans(A3, 0.0, D4) @ _rpy(0.0, 0.0, 0.0) @ _rot_z(q[3])
    # wrist_1 to wrist_2
    T = T @ _trans(0.0, -D5, 0.0) @ _rpy(np.pi / 2.0, 0.0, 0.0) @ _rot_z(q[4])
    # wrist_2 to wrist_3
    T = T @ _trans(0.0, D6, 0.0) @ _rpy(np.pi / 2.0, np.pi, np.pi) @ _rot_z(q[5])
    # flange to tool0
    T_flange = _rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)
    T_tool0 = _rpy(np.pi / 2.0, 0.0, np.pi / 2.0)
    return T @ T_flange @ T_tool0


def solve_ik(
    target_xyz: List[float],
    seed: Optional[List[float]] = None,
    desired_yaw: Optional[float] = None,
) -> Tuple[List[float], float]:
    """
    Giải IK số học tìm cấu hình khớp q sao cho tool0 đạt target_xyz
    với hướng trục Z của tool0 hướng thẳng đứng xuống dưới (-Z của base_link).

    Parameters
    ----------
    target_xyz : List[float]
        Tọa độ mục tiêu [x, y, z] theo hệ base_link (mét).
    seed : Optional[List[float]]
        Cấu hình khớp hiện tại để tối thiểu hóa độ dịch chuyển (tránh lật khớp).
    desired_yaw : Optional[float]
        Góc quay mong muốn của ngón kẹp (radian theo hệ base_link).

    Returns
    -------
    Tuple[List[float], float]
        (q_solution, pos_error_meters)
    """
    target = np.array(target_xyz, dtype=float)
    x0 = np.array(seed if seed is not None else HOME_JOINTS, dtype=float)

    # Khởi tạo góc quay shoulder_pan gần góc phương vị của mục tiêu
    target_pan = np.arctan2(target[1], target[0])
    if abs(target_pan - x0[0]) > np.pi:
        target_pan += 2 * np.pi if x0[0] > target_pan else -2 * np.pi
    x0[0] = 0.5 * (x0[0] + target_pan)

    def cost(q):
        T = fk(q)
        pos = T[:3, 3]
        z_axis = T[:3, 2]  # Trục Z của tool0

        pos_err = np.sum((pos - target) ** 2)
        # Ép trục Z của tool0 hướng thẳng xuống (-Z): z_axis = [0, 0, -1]
        ori_err = (z_axis[0])**2 + (z_axis[1])**2 + (z_axis[2] + 1.0)**2

        # Ràng buộc độ lệch với seed để giữ tính liên tục
        reg = 0.005 * np.sum((q - x0) ** 2)

        return 100.0 * pos_err + 10.0 * ori_err + reg

    # Đảm bảo x0 nằm trong bounds
    for i, (b_min, b_max) in enumerate(UR3E_BOUNDS):
        x0[i] = np.clip(x0[i], b_min + 1e-4, b_max - 1e-4)

    res = minimize(
        cost,
        x0,
        method="SLSQP",
        bounds=UR3E_BOUNDS,
        options={"maxiter": 150, "ftol": 1e-7, "disp": False},
    )

    q_sol = res.x.tolist()

    # Căn chỉnh góc quay wrist_3 sao cho hướng trượt ngón kẹp (trục Y của tool0)
    # hoàn toàn song song hoặc vuông góc với các mặt của khối cube (hoặc theo desired_yaw)
    q_test = list(q_sol)
    q_test[5] = 0.0
    T0 = fk(q_test)
    y_angle_0 = float(np.arctan2(T0[1, 1], T0[0, 1]))
    
    base_yaw = float(desired_yaw) if desired_yaw is not None else 0.0
    candidates = [
        base_yaw,
        base_yaw + np.pi / 2.0,
        base_yaw + np.pi,
        base_yaw - np.pi / 2.0,
    ]
    best_w3 = 0.0
    min_dist = 1e9
    for des in candidates:
        w3 = y_angle_0 - des
        w3 = (w3 + np.pi) % (2.0 * np.pi) - np.pi
        if abs(w3) < min_dist:
            min_dist = abs(w3)
            best_w3 = w3
    q_sol[5] = float(best_w3)

    final_pos = fk(q_sol)[:3, 3]
    pos_error = float(np.linalg.norm(final_pos - target))

    return q_sol, pos_error

