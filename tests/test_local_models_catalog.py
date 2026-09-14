# tests/test_local_models_catalog.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import local_models_catalog as catalog


def test_catalog_contains_expected_models():
    models = catalog.get_curated_models()
    assert len(models) == 3

    model_ids = [m.id for m in models]
    assert "gemma-4-e2b" in model_ids
    assert "gemma-4-12b" in model_ids
    assert "gemma-4-26b-moe" in model_ids


def test_clean_user_facing_names():
    # User requested clean names without 'huihui' or 'abliterated'
    for model in catalog.get_curated_models():
        assert "huihui" not in model.display_name.lower()
        assert "abliterated" not in model.display_name.lower()
        assert len(model.user_description) > 5
        assert model.min_ram_gb > 0
        assert model.download_url.startswith("https://huggingface.co/")
        assert model.context_window == 131072


def test_get_model_by_id():
    m_e2b = catalog.get_model_by_id("gemma-4-e2b")
    assert m_e2b is not None
    assert m_e2b.display_name == "Gemma 4 (E2B)"
    assert m_e2b.min_ram_gb == 4.0

    m_none = catalog.get_model_by_id("non_existent_model")
    assert m_none is None


def test_default_models_dir():
    dir_path = catalog.get_default_models_dir()
    assert isinstance(dir_path, str)
    assert "models" in dir_path

