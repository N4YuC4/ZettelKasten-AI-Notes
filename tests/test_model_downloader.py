# tests/test_model_downloader.py

import os
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from model_downloader import ModelDownloader
from hardware_checker import HardwareChecker
import local_models_catalog as catalog


def test_get_model_path():
    model = catalog.get_model_by_id("gemma-4-e2b")
    path = ModelDownloader.get_model_path(model, "/tmp/models")
    assert path == os.path.join("/tmp/models", model.filename)


def test_is_model_downloaded_false_for_missing():
    model = catalog.get_model_by_id("gemma-4-e2b")
    assert ModelDownloader.is_model_downloaded(model, "/tmp/non_existent_folder_xyz") is False


def test_is_model_downloaded_true_for_existing(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    model_file = tmp_path / model.filename
    # write > 10MB dummy data
    with open(model_file, "wb") as f:
        f.write(b"0" * (11 * 1024 * 1024))

    assert ModelDownloader.is_model_downloaded(model, str(tmp_path)) is True


def test_delete_model(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    model_file = tmp_path / model.filename
    part_file = tmp_path / (model.filename + ".part")
    model_file.write_text("dummy")
    part_file.write_text("dummy")

    assert os.path.exists(model_file)
    assert os.path.exists(part_file)

    deleted = ModelDownloader.delete_model(model, str(tmp_path))
    assert deleted is True
    assert not os.path.exists(model_file)
    assert not os.path.exists(part_file)


def test_download_mocked_success(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    progress_updates = []
    finished_path = []

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "1024"}
    mock_response.iter_content.return_value = [b"a" * 512, b"b" * 512]

    with patch("requests.Session.get", return_value=mock_response):
        ModelDownloader.start_download(
            model=model,
            models_dir=str(tmp_path),
            on_progress=lambda ratio, text, speed: progress_updates.append((ratio, text)),
            on_finished=lambda p: finished_path.append(p)
        )
        time.sleep(0.4)

    assert len(finished_path) == 1
    assert os.path.exists(finished_path[0])
    assert os.path.getsize(finished_path[0]) == 1024


def test_download_mocked_retry_recovery(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    progress_updates = []
    finished_path = []

    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 200
    mock_resp_fail.headers = {"content-length": "2048"}
    
    def fail_generator(*args, **kwargs):
        yield b"x" * 512
        raise ConnectionResetError("SSL connection dropped")
    mock_resp_fail.iter_content.side_effect = fail_generator

    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 206
    mock_resp_success.headers = {"content-length": "1536"}
    mock_resp_success.iter_content.return_value = [b"y" * 1536]

    with patch("requests.Session.get", side_effect=[mock_resp_fail, mock_resp_success]):
        with patch("model_downloader.time.sleep", return_value=None):
            ModelDownloader.start_download(
                model=model,
                models_dir=str(tmp_path),
                on_progress=lambda ratio, text, speed: progress_updates.append((ratio, text)),
                on_finished=lambda p: finished_path.append(p)
            )
            thread = ModelDownloader._active_threads.get(model.id)
            if thread:
                thread.join(timeout=2.0)

    assert len(finished_path) == 1
    assert os.path.exists(finished_path[0])
    assert os.path.getsize(finished_path[0]) == 2048


def test_format_eta():
    from model_downloader import format_eta
    assert format_eta(None) == ""
    assert format_eta(0) == ""
    assert format_eta(45) == " • Kalan: 45 sn"
    assert format_eta(125) == " • Kalan: 2 dk 5 sn"
    assert format_eta(3665) == " • Kalan: 1 sa 1 dk"


def test_status_registry_and_listeners():
    model = catalog.get_model_by_id("gemma-4-e2b")
    received_events = []

    def my_listener(status):
        received_events.append(status.state)

    ModelDownloader.register_listener(model.id, my_listener)
    assert len(received_events) >= 1

    ModelDownloader.unregister_listener(model.id, my_listener)
    init_len = len(received_events)
    ModelDownloader._notify_listeners(model.id)
    assert len(received_events) == init_len


def test_download_insufficient_disk_space(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    error_called = []

    with patch.object(HardwareChecker, "check_disk_space", return_value=(False, "Yetersiz Disk Alanı!", 0.5)):
        ModelDownloader.start_download(
            model=model,
            models_dir=str(tmp_path),
            on_error=lambda err: error_called.append(err)
        )

    assert len(error_called) == 1
    assert "Yetersiz Disk Alanı" in error_called[0]
    status = ModelDownloader.get_status(model.id)
    assert status.state == "error"


def test_download_retry_counter_resets_on_successful_data_transfer(tmp_path):
    model = catalog.get_model_by_id("gemma-4-e2b")
    observed_retry_counts = []
    finished_path = []

    # Attempt 1: Fails to connect (0 bytes)
    mock_resp_fail_init_1 = MagicMock()
    mock_resp_fail_init_1.side_effect = ConnectionError("Initial DNS fail")

    # Attempt 2: Fails to connect (0 bytes)
    mock_resp_fail_init_2 = MagicMock()
    mock_resp_fail_init_2.side_effect = ConnectionError("Initial TCP fail")

    # Attempt 3: Connects, transfers 512 bytes, then connection drops
    mock_resp_partial = MagicMock()
    mock_resp_partial.status_code = 200
    mock_resp_partial.headers = {"content-length": "2048"}
    def partial_gen(*args, **kwargs):
        yield b"p" * 512
        raise ConnectionResetError("Midway SSL failure")
    mock_resp_partial.iter_content.side_effect = partial_gen

    # Attempt 4: Resumes and finishes remaining 1536 bytes
    mock_resp_finish = MagicMock()
    mock_resp_finish.status_code = 206
    mock_resp_finish.headers = {"content-length": "1536"}
    mock_resp_finish.iter_content.return_value = [b"q" * 1536]

    # Hook listener to record retry_counts when state is downloading
    def on_status_change(st):
        observed_retry_counts.append((st.state, st.retry_count, st.status_text))

    ModelDownloader.register_listener(model.id, on_status_change)

    with patch("requests.Session.get", side_effect=[
        mock_resp_fail_init_1.side_effect,
        mock_resp_fail_init_2.side_effect,
        mock_resp_partial,
        mock_resp_finish
    ]):
        with patch("model_downloader.time.sleep", return_value=None):
            ModelDownloader.start_download(
                model=model,
                models_dir=str(tmp_path),
                on_finished=lambda p: finished_path.append(p)
            )
            thread = ModelDownloader._active_threads.get(model.id)
            if thread:
                thread.join(timeout=3.0)

    ModelDownloader.unregister_listener(model.id, on_status_change)

    assert len(finished_path) == 1
    assert os.path.exists(finished_path[0])
    assert os.path.getsize(finished_path[0]) == 2048

    # Verify that when midway failure occurred after streaming data, retry_count reset and became 1, not 3!
    retry_statuses = [text for state, count, text in observed_retry_counts if "Bağlantı koptu" in text]
    assert len(retry_statuses) >= 3
    # First failure -> (1/30)
    assert "(1/30)" in retry_statuses[0]
    # Second failure -> (2/30)
    assert "(2/30)" in retry_statuses[1]
    # Third failure (after streaming 512 bytes): MUST have reset so it is (1/30), NOT (3/30)!
    assert "(1/30)" in retry_statuses[2]




