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
    # Check if logs directory creation is handled
    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_setup_logging_directory_creation() -> None:
    """Test that setup_logging creates the directory if it doesn't exist."""
    # We patch the Path object within the logger module
    with patch("coreason_inference.utils.logger.Path") as MockPath:
        # Mock Path instance
        mock_path_instance = MagicMock()
        MockPath.return_value = mock_path_instance

        # Scenario 1: Directory does not exist
        mock_path_instance.exists.return_value = False

        # Call setup_logging
        setup_logging()

        # Verify mkdir was called
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

        # Scenario 2: Directory exists
        mock_path_instance.exists.return_value = True
        mock_path_instance.mkdir.reset_mock()

        setup_logging()
        mock_path_instance.mkdir.assert_not_called()
