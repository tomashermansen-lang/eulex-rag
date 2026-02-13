import pytest

from src.common.config_loader import load_settings, clear_config_cache
import src.common.config_loader as config_loader


def test_load_settings_validates_retrieval_pool_size(monkeypatch, tmp_path):
    """Test that invalid retrieval_pool_size value raises ValueError."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "settings.yaml"
    config_path.write_text(
        """
rag:
  retrieval_pool_size: 0
""".lstrip(),
        encoding="utf-8",
    )

    # Patch the config dir and clear cache
    monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
    clear_config_cache()

    with pytest.raises(ValueError, match=r"rag\.retrieval_pool_size must be >= 1"):
        load_settings()


def test_load_settings_validates_eurlex_overlap(monkeypatch, tmp_path):
    """Test that eurlex overlap >= chunk_tokens raises ValueError."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "settings.yaml"
    config_path.write_text(
        """
eurlex:
  chunk_tokens: 100
  overlap: 100
""".lstrip(),
        encoding="utf-8",
    )

    # Patch the config dir and clear cache
    monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
    clear_config_cache()

    with pytest.raises(ValueError, match=r"eurlex\.overlap must be < eurlex\.chunk_tokens"):
        load_settings()
