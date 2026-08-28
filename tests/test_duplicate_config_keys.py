"""A duplicate YAML key ate a deploy, silently, on 28 August.

Sixteen dead markets were added to `instruments.ignored_symbols` to get them
out of the scan. Ninety lines further down the same mapping there was already
an `ignored_symbols: []`. YAML's rule is last-one-wins, so the empty list won
and the scan was unchanged.

The change was complete, correct, committed, pulled and deployed. The operator
ran `update.cmd`, restarted, and watched exactly the same markets refused in
exactly the same way. Nothing in the loader, the tests, the launcher or the log
said a word, because as far as YAML is concerned nothing went wrong.

THAT IS THE DEFECT CLASS THIS ACCOUNT KEEPS PRODUCING -- a change that exists,
is correct, and never reaches the path the code takes -- and here the config
format itself is the mechanism. A 3,600-line file that governs real money
cannot have an edit-shaped hole in it, so duplicate keys are now a hard error
that names both line numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import DEFAULT_CONFIG_PATH, ConfigError, _read_yaml, load_settings


def _yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestADuplicateKeyIsRefused:
    def test_the_exact_shape_that_ate_the_deploy(self, tmp_path: Path) -> None:
        path = _yaml(
            tmp_path,
            "instruments:\n"
            "  ignored_symbols: [CORN, SG30]\n"
            "  symbol_suffix: .i\n"
            "  ignored_symbols: []\n",
        )

        with pytest.raises(ConfigError, match="duplicate key 'ignored_symbols'"):
            _read_yaml(path)

    def test_the_error_names_both_lines_so_the_fix_is_obvious(self, tmp_path: Path) -> None:
        """A message saying only "duplicate key" sends someone hunting through
        3,600 lines for the other one. Ninety lines apart is the realistic
        distance, and it is exactly the distance at which you do not find it by
        looking."""
        path = _yaml(tmp_path, "a:\n  k: 1\n  b: 2\n  c: 3\n  k: 4\n")

        with pytest.raises(ConfigError) as caught:
            _read_yaml(path)

        assert "line 2" in str(caught.value)
        assert "line 5" in str(caught.value)

    def test_a_duplicate_at_the_top_level_is_caught_too(self, tmp_path: Path) -> None:
        path = _yaml(tmp_path, "risk:\n  a: 1\nfilters:\n  b: 2\nrisk:\n  a: 9\n")

        with pytest.raises(ConfigError, match="duplicate key 'risk'"):
            _read_yaml(path)

    def test_a_duplicate_nested_several_levels_down(self, tmp_path: Path) -> None:
        """The overlay nests four deep in places. A check that only looked at
        the top level would have missed the one that actually happened."""
        path = _yaml(tmp_path, "a:\n  b:\n    c:\n      d: 1\n      e: 2\n      d: 3\n")

        with pytest.raises(ConfigError, match="duplicate key 'd'"):
            _read_yaml(path)


class TestEverythingLegalStillLoads:
    def test_the_same_key_in_two_different_mappings_is_fine(self, tmp_path: Path) -> None:
        """`enabled: true` appears dozens of times in the real overlay. Only a
        repeat inside ONE mapping is the error."""
        path = _yaml(tmp_path, "a:\n  enabled: true\nb:\n  enabled: false\n")

        assert _read_yaml(path) == {"a": {"enabled": True}, "b": {"enabled": False}}

    def test_the_shipped_configs_load(self) -> None:
        """The check is worthless if it had to be relaxed to let the account's
        own config through -- and it is exactly the kind of check that gets
        relaxed rather than obeyed."""
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.instruments.symbol_suffix == ".i"


class TestTheTwoListsAreNotTheSameThing:
    """The mistake the duplicate key was hiding, kept as a standing fact.

    Sixteen dead markets were put on `blocklist` to get them out of the scan.
    `blocklist` refuses an ENTRY -- the symbol is still fetched, analysed and
    run through the whole filter chain, and only at the end does the answer
    come back no. `ignored_symbols` is the one that removes it from the
    universe. The schema docstring states the difference plainly; it was read,
    quoted in the config comment as the REASON for choosing `blocklist`, and
    the property being quoted was the one that made it unfit.

    THE OWNER THEN CHOSE TO KEEP EVERY MARKET IN THE SCAN -- "gwn laten" -- so
    both lists are back to what they were. These tests therefore pin the
    DISTINCTION and not any particular list, because the distinction is what
    was misunderstood and the list is his call.
    """

    def test_a_blocklisted_symbol_is_still_scanned(self) -> None:
        instruments = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).instruments

        assert "UKOUSD" in instruments.blocklist
        # Refused an entry, and still in the universe. That is the whole
        # difference, and it is why the first attempt changed nothing.
        assert not instruments.is_ignored("UKOUSD")

    def test_only_the_ignore_list_reaches_the_scanner(self) -> None:
        """Asserted over the source because the two are one word apart at the
        call site, and picking the wrong one produces a config that looks
        applied, tests green, and does nothing."""
        import inspect

        from scanner import universe

        source = " ".join(inspect.getsource(universe).split())

        assert "is_ignored(item.name)" in source
        assert "blocklist" not in source

    def test_oil_is_on_the_blocklist_and_that_is_the_right_list_for_it(self) -> None:
        """UKOUSD and USOUSD have four real trades behind them, and
        `ignored_symbols` drops a symbol out of position management and
        emergency flattening as well. A live oil ticket would stop being
        managed. For a market that has held a position, `blocklist` is the
        correct instrument and `ignored_symbols` would be dangerous."""
        instruments = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).instruments

        for symbol in ("UKOUSD", "USOUSD"):
            assert symbol in instruments.blocklist
            assert not instruments.is_ignored(symbol)
