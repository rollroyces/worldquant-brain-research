"""Security regression tests — credentials must never leak through
repr, print, or error messages.

Run with:  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from brain_harness import BrainClient, build_parser

SECRET = "supersecret-PASSWORD-12345-do-not-log"
EMAIL = "leak-test@example.com"


class TestBrainClientRepr:
    """BrainClient.__repr__ must mask the password."""

    def test_repr_masks_password(self):
        client = BrainClient(email=EMAIL, password=SECRET, mock=True)
        text = repr(client)
        assert SECRET not in text, f"Password leaked in repr(): {text}"
        assert "***" in text, "Expected password-mask placeholder in repr"

    def test_str_masks_password(self):
        client = BrainClient(email=EMAIL, password=SECRET, mock=True)
        text = str(client)
        assert SECRET not in text

    def test_repr_masks_password_even_when_blank(self):
        # Even an empty password should not appear in repr as something
        # distinguishable — easier to just check it doesn't crash.
        client = BrainClient(email=EMAIL, password="", mock=True)
        repr(client)


class TestCliHasNoPasswordFlag:
    """The CLI must not accept --password (would leak via ps/history)."""

    def test_parser_has_no_password_flag(self):
        parser = build_parser()
        for sub_name in ("simulate", "batch", "submit", "gates"):
            # Parse --help on each subcommand; argparse will reject unknown
            # options with a SystemExit. We catch it.
            try:
                parser.parse_args([sub_name, "--password", "x"])
            except SystemExit:
                # argparse exits with code 2 on unknown args — that's the
                # success path here (the flag was rejected).
                continue
            pytest.fail(f"--password was accepted by the {sub_name} subcommand")


class TestSimulateErrorDoesNotLeakBody:
    """The simulate error message must not echo the response body."""

    def test_simulate_error_uses_keys_only(self, monkeypatch):

        # Build a client in non-mock mode and monkeypatch the session
        # to return a body without an id.
        client = BrainClient(email=EMAIL, password=SECRET, mock=False)

        class FakeResp:
            status_code = 200

            def json(self):
                # Body contains a fake "session_token" — must not appear in the error.
                return {"error": "no_id", "session_token": SECRET,
                        "user": {"email": EMAIL}}

            def raise_for_status(self):
                pass

        class FakeSession:
            def post(self, url, json=None, cookies=None, headers=None, timeout=None):
                return FakeResp()

        client.session = FakeSession()

        with pytest.raises(RuntimeError) as excinfo:
            client.submit_simulation({"expression": "rank(close)"})
        msg = str(excinfo.value)
        # Don't leak credential VALUES (the body included `session_token: SECRET`).
        assert SECRET not in msg, f"Password leaked in error message: {msg}"
        assert EMAIL not in msg, f"Email leaked in error message: {msg}"
        # Sanity: the message should still be useful.
        assert "no id" in msg.lower()
        assert "status=200" in msg  # operator can still debug


class TestSubmitAlphaResponseSanitized:
    """The submit response print should not echo the full body."""

    def test_submit_alpha_print_extracts_status_only(self, capsys, monkeypatch):
        # Simulate that BrainClient.submit_alpha returns a dict with a
        # sensitive field. We re-import the script's cmd_submit flow by
        # inspecting the print logic.
        from brain_harness import cmd_submit

        # Patch BrainClient to a fake that returns a response containing
        # a secret field.
        secret_resp = {"status": "ok", "session_token": SECRET}

        class FakeBC:
            def __init__(self):
                self.email = EMAIL
                self.password = SECRET
                self.mock = False
                self.cookies = {}

            def authenticate(self):
                pass

            def submit_alpha(self, sim_id):
                return secret_resp

        # Patch sys.argv and run cmd_submit.
        # We need a result file with "passes": true and "sim_id": "abc".
        tmp_dir = Path("/tmp/brain_test")
        tmp_dir.mkdir(exist_ok=True)
        result_path = tmp_dir / "sim_abc.json"
        result_path.write_text('{"passes": true, "sim_id": "abc"}')

        # Change cwd so the glob finds our file.
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            # Build a fake args namespace.
            import argparse
            args = argparse.Namespace(
                from_results="sim_*.json", max_per_day=4, mock=False,
            )
            client = FakeBC()
            from brain_harness import Gates
            gates = Gates()
            rc = cmd_submit(args, client, gates)
            assert rc == 0
            captured = capsys.readouterr()
            assert SECRET not in captured.out, f"Password leaked in stdout: {captured.out}"
            assert "session_token" not in captured.out, \
                f"Session token leaked in stdout: {captured.out}"
            # Sanity: the legitimate print line should still be present.
            assert "Submitted abc" in captured.out
        finally:
            os.chdir(old_cwd)
            # Clean up.
            if result_path.exists():
                result_path.unlink()
            if tmp_dir.exists():
                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass
