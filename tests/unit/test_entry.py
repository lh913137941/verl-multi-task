import os
from pathlib import Path


def test_native_source_root_is_optional_for_control_plane_unit_suite():
    root = os.environ.get("MT_VERL_SOURCE_ROOT")
    if root:
        assert Path(root).is_absolute()
