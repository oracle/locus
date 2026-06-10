# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Built-in tools that most agents end up needing.

Drop these into an agent's tool list to give the model common primitives
without having to re-implement them per product.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from tulip.tools.decorator import tool


@tool(idempotent=True)
def get_today_date() -> dict:
    """Return today's date plus common reference points for date arithmetic.

    Call this whenever the user mentions a relative or partial date
    ("tomorrow", "next Monday", "in ten days", "April 20") so you can
    convert to an explicit YYYY-MM-DD before calling a date-sensitive tool.

    Returns:
        A dict with:

        - ``today`` — today's date (YYYY-MM-DD)
        - ``weekday`` — e.g. ``"Saturday"``
        - ``year`` — current year
        - ``tomorrow`` / ``day_after_tomorrow``
        - ``next_7_days_by_weekday`` — map of lower-cased weekday → ISO date
            for the next seven days, so "Monday" / "Friday" resolve without
            further arithmetic
        - ``one_week_from_now`` / ``two_weeks_from_now``
    """
    now = datetime.now().astimezone()
    today = now.date()
    return {
        "today": today.isoformat(),
        "weekday": now.strftime("%A"),
        "year": today.year,
        "tomorrow": (today + timedelta(days=1)).isoformat(),
        "day_after_tomorrow": (today + timedelta(days=2)).isoformat(),
        "next_7_days_by_weekday": {
            (today + timedelta(days=n)).strftime("%A").lower(): (
                today + timedelta(days=n)
            ).isoformat()
            for n in range(1, 8)
        },
        "one_week_from_now": (today + timedelta(days=7)).isoformat(),
        "two_weeks_from_now": (today + timedelta(days=14)).isoformat(),
    }


__all__ = ["get_today_date"]
