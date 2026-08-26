"""A release moves more than the instruments with that currency in the name.

THE OWNER PUT IT PLAINLY: do not trade around a red folder "anywhere it has
influence", not only on the pair that happens to carry the currency.

Half of that was already fixed and cost real money to learn. `symbol_currencies`
used to return exactly the two legs, so a live XAUAUD short watched Australia
while it waited on American inflation, and a red-folder release took the stop
for -1.01R -- the largest single loss of that day. Metals and crypto now add USD
because they are dollar-denominated wherever they trade.

THE SAME HOLE, ONE FAMILY OVER. That repair keys on the BASE, so equity indices
fell straight through it. GER40 is quoted in euros and has no dollar leg
anywhere, and an FOMC decision or a payrolls print moves it about as hard as it
moves the S&P.

It mattered little while indices sat at the back of the scan rotation. Section
six now has `index` in its every-cycle lane, so GER40, UK100, FRA40 and JP225
are live candidates on the minute a US release lands -- and the calendar would
have been asked about euros, pounds and yen only.
"""

from __future__ import annotations

from filters.calendar.events import symbol_currencies


class TestAnIndexTradesUSDataWhateverItIsQuotedIn:
    def test_a_euro_index_is_watched_against_american_releases(self) -> None:
        assert "USD" in symbol_currencies("EUR", "EUR", "index")

    def test_the_same_pair_that_is_not_an_index_is_not(self) -> None:
        """The addition has to be about the FAMILY, not about euros. If this
        also fired for spot FX the blackout would swallow every pair on the
        board and the filter would stop meaning anything."""
        assert "USD" not in symbol_currencies("EUR", "GBP", "forex")

    def test_every_index_in_the_catalogue_is_covered_not_just_the_us_ones(self) -> None:
        """US30 and NAS100 were always covered -- they are quoted in dollars.
        The ones that were not are exactly the ones section six just gained
        access to."""
        for base, quote in (("EUR", "EUR"), ("GBP", "GBP"), ("JPY", "JPY"), ("AUD", "AUD")):
            assert "USD" in symbol_currencies(base, quote, "index"), f"{base}{quote}"

    def test_the_home_currency_is_still_watched_too(self) -> None:
        """Adding USD must not replace the local calendar. A German index is
        moved by the ECB as well as by the Fed."""
        currencies = symbol_currencies("EUR", "EUR", "index")

        assert {"EUR", "USD"} <= currencies


class TestNothingElseChanged:
    def test_the_gold_repair_still_holds(self) -> None:
        """The XAUAUD loss that started all of this. Australia was watched;
        American inflation was what moved it."""
        assert "USD" in symbol_currencies("XAU", "AUD")

    def test_a_caller_without_a_spec_behaves_exactly_as_before(self) -> None:
        """`asset_class` is optional so that a call site which has no spec to
        hand keeps its old answer rather than being silently loosened."""
        assert symbol_currencies("EUR", "GBP") == symbol_currencies("EUR", "GBP", None)
        assert symbol_currencies("EUR", "GBP") == frozenset({"EUR", "GBP"})

    def test_an_unknown_asset_class_adds_nothing(self) -> None:
        assert symbol_currencies("EUR", "GBP", "something_new") == frozenset({"EUR", "GBP"})


class TestTheWindowsThemselves:
    """The owner asked for 30 minutes either side of a red folder. What is
    shipped is stricter, and this records the actual numbers so a later edit
    that quietly shortens them fails here."""

    def config(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_a_red_folder_is_two_hours_of_silence_before_and_one_after(self) -> None:
        extreme = self.config().filters.news.extreme_impact

        assert extreme.minutes_before >= 30
        assert extreme.minutes_after >= 30
        assert (extreme.minutes_before, extreme.minutes_after) == (120, 60)

    def test_an_ordinary_high_impact_release_clears_the_asked_for_thirty(self) -> None:
        high = self.config().filters.news.high_impact

        assert high.minutes_before >= 30
        assert high.minutes_after >= 30

    def test_the_releases_that_count_as_extreme_are_the_red_folders(self) -> None:
        keywords = {word.lower() for word in self.config().filters.news.extreme_keywords}

        for expected in ("fomc", "nfp", "cpi", "interest rate decision"):
            assert any(expected in word for word in keywords), expected

    def test_section_six_carries_its_own_margin_on_top(self) -> None:
        """A second sieve, and at 15 it could not catch anything the 60-minute
        filter above already let through. At 30 it is at least capable of
        refusing something."""
        assert self.config().analysis.candle_momentum.news_clearance_minutes >= 30.0

    def test_no_calendar_still_means_no_trade(self) -> None:
        """The owner's standing rule, and the one that must never be traded
        away for more setups."""
        assert self.config().filters.news.fail_closed
