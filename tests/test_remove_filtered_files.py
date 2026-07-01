"""Regression tests for the NFC/NFD unlink fallback in ``remove-filtered-files``.

The data drive is exFAT via macOS FSKit and does no filename normalization, so a
file is removable only by the exact bytes it was created with. ``iterdir`` hands
those entries back in NFD, but a yt-dlp video may be stored in NFC, so a direct
``unlink`` of the discovered (NFD) path raises ``ENOENT`` and the code retries
with the NFC form.

The test machine's own filesystem (APFS, normalization-insensitive) resolves NFC
and NFD to the same directory entry and so never exercises the fallback, and
Linux CI (exact-byte) would behave differently again. Both are therefore useless
for pinning this behavior. ``_ExactBytePath`` models exFAT's exact-byte
semantics against an in-memory set so the tests are deterministic everywhere.
"""

from __future__ import annotations

import importlib.util
import sys
import unicodedata
from pathlib import Path
from typing import Any, cast

import pytest


def _load_script_module() -> Any:
    script_path = Path(__file__).parent.parent / "scripts" / "remove-filtered-files.py"
    spec = importlib.util.spec_from_file_location("remove_filtered_files", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Any, module)


class _ExactBytePath:
    """Minimal ``Path`` stand-in with exFAT-style exact-byte unlink semantics.

    ``disk`` is the shared set of names physically present, keyed by their exact
    stored bytes. ``unlink`` matches ``name`` literally -- no normalization -- so
    an NFD lookup misses an NFC-stored file and vice versa, which is precisely
    what a real ``Path`` cannot reproduce on the test host.
    """

    def __init__(self, name: str, disk: set[str]) -> None:
        self._name = name
        self._disk = disk

    @property
    def name(self) -> str:
        return self._name

    def with_name(self, name: str) -> _ExactBytePath:
        return _ExactBytePath(name, self._disk)

    def unlink(self) -> None:
        try:
            self._disk.remove(self._name)
        except KeyError as exc:
            raise FileNotFoundError(2, "No such file or directory", self._name) from exc

    def __str__(self) -> str:
        return self._name


def _norm_pair() -> tuple[str, str]:
    """Return an ``(nfc, nfd)`` filename pair whose two forms differ on disk."""
    base = "Café [vid123].wav"
    nfc = unicodedata.normalize("NFC", base)
    nfd = unicodedata.normalize("NFD", base)
    assert nfc != nfd  # guards the premise: the é must actually re-encode
    return nfc, nfd


def test_unlink_deletes_discovered_nfd_file_without_touching_nfc_form() -> None:
    module = _load_script_module()
    nfc, nfd = _norm_pair()
    disk = {nfd, nfc}

    module._unlink_either_norm(_ExactBytePath(nfd, disk))

    # The direct unlink succeeds, so the NFC fallback never runs and its file survives.
    assert disk == {nfc}


def test_unlink_falls_back_to_nfc_when_file_stored_as_nfc() -> None:
    module = _load_script_module()
    nfc, nfd = _norm_pair()
    disk = {nfc}

    # iterdir handed back the NFD form, but the file is stored NFC (the yt-dlp case).
    module._unlink_either_norm(_ExactBytePath(nfd, disk))

    assert disk == set()


def test_unlink_raises_for_stale_target_when_neither_form_present() -> None:
    module = _load_script_module()
    _, nfd = _norm_pair()
    disk: set[str] = set()

    # Concurrent cleanup / stale target list: both forms are gone, so the fallback
    # re-raises instead of silently succeeding -- nothing is deleted.
    with pytest.raises(FileNotFoundError):
        module._unlink_either_norm(_ExactBytePath(nfd, disk))
    assert disk == set()


def test_unlink_fallback_never_touches_a_differently_named_sibling() -> None:
    module = _load_script_module()
    _, nfd = _norm_pair()
    unrelated = "Unrelated [other99].wav"
    disk = {unrelated}

    # The stale target's file is gone and only an unrelated file remains. The
    # fallback retries the target's OWN NFC name and nothing else, so it cannot
    # over-delete a sibling: it raises and leaves the unrelated file intact.
    with pytest.raises(FileNotFoundError):
        module._unlink_either_norm(_ExactBytePath(nfd, disk))
    assert disk == {unrelated}


def test_delete_targets_records_stale_target_as_failure_and_leaves_others(capsys: Any) -> None:
    module = _load_script_module()
    _, nfd = _norm_pair()
    survivor = "Keep me [keep01].wav"
    disk = {survivor}
    targets = [("data_downloads_audio_dir", "Channel", "vid123", _ExactBytePath(nfd, disk))]

    deleted, failed = module._delete_targets(targets)

    # A vanished target is counted as a failure (resilient processing) and no
    # other on-disk file is disturbed.
    assert (deleted, failed) == (0, 1)
    assert disk == {survivor}
    assert "FAIL:" in capsys.readouterr().err


def test_delete_targets_deletes_nfc_stored_file_through_the_fallback() -> None:
    module = _load_script_module()
    nfc, nfd = _norm_pair()
    disk = {nfc}
    targets = [("data_downloads_audio_dir", "Channel", "vid123", _ExactBytePath(nfd, disk))]

    deleted, failed = module._delete_targets(targets)

    assert (deleted, failed) == (1, 0)
    assert disk == set()
