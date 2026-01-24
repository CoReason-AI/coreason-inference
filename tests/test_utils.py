# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_inference.utils.logger import logger, setup_logging


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_setup_logging_creates_directory() -> None:
    """Test that setup_logging creates the directory if it doesn't exist."""
    with patch("coreason_inference.utils.logger.Path") as MockPath:
        mock_path_instance = MagicMock()
        MockPath.return_value = mock_path_instance
        # Simulate directory does not exist
        mock_path_instance.exists.return_value = False

        setup_logging()

        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
