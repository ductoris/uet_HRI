# conftest.py — Cấu hình pytest cho ur3_llm_control
# Vô hiệu hóa plugin launch_testing_ros không tương thích với pytest>=9
collect_ignore_glob = []


def pytest_configure(config):
    """Disable ROS launch_testing plugin that conflicts with newer pytest."""
    try:
        config.pluginmanager.set_blocked("launch_testing_ros_pytest_entrypoint")
    except Exception:
        pass
    try:
        config.pluginmanager.set_blocked("launch_ros")
    except Exception:
        pass
