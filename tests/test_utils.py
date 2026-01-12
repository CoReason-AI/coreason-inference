# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from unittest.mock import patch

from coreason_inference.utils.logger import setup_logging


def test_logger_initialization() -> None:
    with patch("coreason_inference.utils.logger.logger") as mock_logger:
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                # Case 1: Directory does not exist
                mock_exists.return_value = False
                setup_logging()
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

                # Verify logger configuration
                assert mock_logger.remove.called
                assert mock_logger.add.call_count >= 2


def test_logger_directory_exists() -> None:
    with patch("coreason_inference.utils.logger.logger") as mock_logger:
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                # Case 2: Directory exists
                mock_exists.return_value = True
                setup_logging()
                mock_mkdir.assert_not_called()

                # We need to verify side effects or mock calls to satisfy coverage if necessary,
                # but mainly we want to avoid unused variable errors.
                assert mock_logger.remove.called
