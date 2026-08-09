"""Writing a secret must not corrupt the file that holds every other one.

`config/.env` carries the MT5 password, the Claude key and now the database
DSN. A writer that appends instead of replacing, or that drops the lines it did
not recognise, breaks the account's ability to log in — and it breaks it at the
next restart rather than immediately, which is the worst time to find out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("set_secret", ROOT / "scripts" / "set_secret.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module():  # type: ignore[no-untyped-def]
    return load_module()


class TestWritingASecret:
    def test_it_creates_the_file_when_there_is_none(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "config" / ".env"

        module.write_secret("NEON_DATABASE_URL", "postgresql://u:p@host/db", path)

        assert path.read_text() == "NEON_DATABASE_URL=postgresql://u:p@host/db\n"

    def test_it_keeps_every_other_secret(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """The failure that matters. Losing MT5_PASSWORD here means the account
        cannot log in, and it surfaces at the next restart rather than now."""
        path = tmp_path / ".env"
        path.write_text("MT5_LOGIN=5049535\nMT5_PASSWORD=secret\nMT5_SERVER=Eightcap-Live\n")

        module.write_secret("NEON_DATABASE_URL", "postgresql://u:p@host/db", path)

        lines = path.read_text().splitlines()
        assert "MT5_LOGIN=5049535" in lines
        assert "MT5_PASSWORD=secret" in lines
        assert "MT5_SERVER=Eightcap-Live" in lines
        assert "NEON_DATABASE_URL=postgresql://u:p@host/db" in lines

    def test_it_replaces_rather_than_appending(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """`python-dotenv` takes the last occurrence, so a duplicated key is a
        configuration whose meaning depends on where in the file you look."""
        path = tmp_path / ".env"
        path.write_text("NEON_DATABASE_URL=old\nMT5_LOGIN=1\n")

        module.write_secret("NEON_DATABASE_URL", "new", path)

        lines = path.read_text().splitlines()
        assert lines.count("NEON_DATABASE_URL=new") == 1
        assert "NEON_DATABASE_URL=old" not in lines

    def test_comments_and_blank_lines_survive(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """The file is documentation as much as configuration."""
        path = tmp_path / ".env"
        path.write_text("# MetaTrader\nMT5_LOGIN=1\n\n# Alerts\n# TELEGRAM_BOT_TOKEN=\n")

        module.write_secret("ANTHROPIC_API_KEY", "sk-ant-x", path)

        text = path.read_text()
        assert "# MetaTrader" in text
        assert "# Alerts" in text
        assert "# TELEGRAM_BOT_TOKEN=" in text

    def test_a_commented_out_key_is_not_treated_as_set(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """`# NEON_DATABASE_URL=` is the template's placeholder, not a value.
        Removing it would silently edit the documentation."""
        path = tmp_path / ".env"
        path.write_text("# NEON_DATABASE_URL=postgresql://USER:PASSWORD@HOST/neondb\n")

        module.write_secret("NEON_DATABASE_URL", "real", path)

        lines = path.read_text().splitlines()
        assert "# NEON_DATABASE_URL=postgresql://USER:PASSWORD@HOST/neondb" in lines
        assert "NEON_DATABASE_URL=real" in lines

    def test_the_file_ends_with_a_newline(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """Without it the next appended key joins onto this one's value."""
        path = tmp_path / ".env"
        path.write_text("MT5_LOGIN=1")

        module.write_secret("MT5_SERVER", "Eightcap-Live", path)

        assert path.read_text().endswith("\n")
        assert "MT5_LOGIN=1" in path.read_text().splitlines()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
    def test_it_is_readable_only_by_its_owner(self, module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / ".env"

        module.write_secret("MT5_PASSWORD", "secret", path)

        assert oct(path.stat().st_mode)[-3:] == "600"


class TestTheKeyMustBeOneWeRecognise:
    def test_a_typo_is_refused_rather_than_written(self, module, capsys) -> None:  # type: ignore[no-untyped-def]
        """A misspelled key produces a variable nothing reads, and the failure
        it causes ('the brain is not connected') points nowhere near it."""
        assert module.main(["NEON_DATABSE_URL"]) == 1
        assert "Unknown key" in capsys.readouterr().out

    def test_no_argument_lists_what_is_available(self, module, capsys) -> None:  # type: ignore[no-untyped-def]
        assert module.main([]) == 1
        assert "NEON_DATABASE_URL" in capsys.readouterr().out

    def test_the_dsn_key_matches_what_the_brain_reads(self, module) -> None:
        """Two places naming the same variable is two places to typo it."""
        from brain.store import DSN_ENV

        assert DSN_ENV in module.KNOWN


class TestItNeverEchoesTheValue:
    def test_the_prompt_uses_getpass(self, module, monkeypatch, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        """Typed input must not reach the terminal, the scrollback, or -- on
        Windows -- the console host buffer that outlives the window."""
        monkeypatch.setattr(module, "ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(module, "getpass", lambda _prompt: "postgresql://u:hunter2@h/db")

        assert module.main(["NEON_DATABASE_URL"]) == 0

        out = capsys.readouterr().out
        assert "hunter2" not in out
        assert "postgresql://u:hunter2@h/db" not in out

    def test_it_confirms_the_shape_without_the_secret(  # type: ignore[no-untyped-def]
        self, module, monkeypatch, tmp_path, capsys
    ) -> None:
        """Enough to prove the right thing landed, not enough to be worth a
        screenshot."""
        monkeypatch.setattr(module, "ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(module, "getpass", lambda _prompt: "x" * 60)

        module.main(["ANTHROPIC_API_KEY"])

        out = capsys.readouterr().out
        assert "60 characters" in out
        assert "x" * 60 not in out

    def test_an_empty_answer_leaves_the_file_alone(self, module, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / ".env"
        path.write_text("MT5_LOGIN=1\n")
        monkeypatch.setattr(module, "ENV_PATH", path)
        monkeypatch.setattr(module, "getpass", lambda _prompt: "   ")

        assert module.main(["MT5_PASSWORD"]) == 1
        assert path.read_text() == "MT5_LOGIN=1\n"


class TestTheSuiteNeverTouchesTheRealEnvFile:
    """This is not hypothetical. An earlier version of `write_secret` bound
    `ENV_PATH` as a default argument, which captures it at import time, so
    monkeypatching the module global had no effect and `main()` wrote the test
    fixtures straight into the operator's own config/.env — `hunter2` and sixty
    x's, on top of the MT5 password and the Claude key.

    On a developer machine that is an annoyance. On the VPS, running the suite
    would destroy the credentials the account logs in with, and the failure
    would not appear until the next restart.
    """

    def test_write_secret_resolves_its_path_at_call_time(  # type: ignore[no-untyped-def]
        self, module, monkeypatch, tmp_path
    ) -> None:
        """The specific defect: a default argument would still point at the
        real file no matter what the test set."""
        target = tmp_path / "redirected.env"
        monkeypatch.setattr(module, "ENV_PATH", target)

        module.write_secret("MT5_SERVER", "Eightcap-Live")

        assert target.exists(), "the module global has to be honoured"
        assert "Eightcap-Live" in target.read_text()

    def test_the_default_is_not_frozen_into_the_signature(self, module) -> None:
        """Read from the signature directly, because the behavioural test above
        passes for the wrong reason if someone reintroduces the default and
        happens to import before the patch."""
        import inspect

        default = inspect.signature(module.write_secret).parameters["path"].default

        assert default is None, "path must default to None and resolve inside the function"

    def test_running_main_writes_only_where_it_was_pointed(  # type: ignore[no-untyped-def]
        self, module, monkeypatch, tmp_path
    ) -> None:
        """End to end, through the entry point that caused the damage."""
        from pathlib import Path

        real = Path(module.ROOT) / "config" / ".env"
        before = real.read_bytes() if real.exists() else None

        monkeypatch.setattr(module, "ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(module, "getpass", lambda _prompt: "postgresql://u:p@h/db")
        module.main(["NEON_DATABASE_URL"])

        after = real.read_bytes() if real.exists() else None
        assert after == before, "the real config/.env was modified by a test"
        assert (tmp_path / ".env").exists()
