# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a shell line runs — the facts a command gate decides on.

Every case here is a line a regex over the raw text gets wrong in one
direction or the other: a dangerous word that is only quoted, or a dangerous
program hiding behind a harmless first word.
"""

from __future__ import annotations

import pytest

from tulip.control.shell import parse_command


def _programs(line: str) -> list[str]:
    return [c.name for c in parse_command(line).commands if c.argv]


@pytest.mark.parametrize(
    "line",
    [
        "pytest -k shutdown",
        'git commit -m "handle shutdown and reboot"',
        "grep -rn 'rm -rf' src/",
        "echo rm -rf /",
        "ls # rm -rf /",
    ],
)
def test_a_quoted_or_argument_word_is_not_a_program(line: str) -> None:
    programs = _programs(line)
    assert "rm" not in programs
    assert "shutdown" not in programs
    assert "reboot" not in programs


@pytest.mark.parametrize(
    ("line", "hidden"),
    [
        ("find . -name '*.pyc' -exec rm -rf {} \\;", "rm"),
        ("find . -execdir shred {} +", "shred"),
        ("env rm -rf build", "rm"),
        ("sudo -u root rm -fr ~", "rm"),
        ("FOO=1 timeout -s KILL 5 reboot", "reboot"),
        ("nice -n 10 nohup shutdown now", "shutdown"),
        ("ls\nrm -rf /", "rm"),
        ("ls; rm -rf /", "rm"),
        ("ls && rm -rf /", "rm"),
        ("(cd x && rm -rf y)", "rm"),
        ("echo $(rm -rf /)", "rm"),
        ('echo "$(rm -rf /)"', "rm"),
        ("echo `reboot`", "reboot"),
        ("diff <(reboot) b", "reboot"),
        ("ls | xargs -n1 -I {} rm -rf {}", "rm"),
        ("bash -c 'rm -rf /'", "rm"),
        ("sh -lc 'cd /; rm -rf x'", "rm"),
        ("eval rm -rf /", "rm"),
        ("echo a#b; rm -rf /", "rm"),
        ("/bin/rm -rf /", "rm"),
        ("cat <<EOF > f\n$(reboot)\nEOF", "reboot"),
        ("find . -exec sh -c 'rm -rf x' \\;", "rm"),
        ("ls | xargs sh -c 'reboot'", "reboot"),
    ],
)
def test_a_program_behind_another_is_found(line: str, hidden: str) -> None:
    assert hidden in _programs(line)


def test_a_quoted_heredoc_body_runs_nothing() -> None:
    parsed = parse_command("cat <<'EOF' > notes.md\n$(reboot)\nshutdown now\nEOF\nls")
    assert _programs(parsed.source) == ["cat", "ls"]
    assert ("<<", "EOF") in parsed.commands[0].redirects


def test_a_tab_stripped_heredoc_ends_on_an_indented_delimiter() -> None:
    parsed = parse_command("cat <<-END\n\treboot\n\tEND\nls")
    assert _programs(parsed.source) == ["cat", "ls"]


def test_wrappers_and_origins_are_recorded() -> None:
    parsed = parse_command("sudo env A=1 rm x; find . -exec cat {} +; echo $(date)")
    rm, find, cat, echo, date = parsed.commands
    assert rm.wrappers == ("sudo", "env")
    assert rm.argv == ("rm", "x")
    assert find.origin == ""
    assert cat.origin == "exec"
    assert echo.connector == ";"
    assert date.origin == "substitution"
    assert date.nested
    assert parsed.has_substitution
    assert [c.name for c in parsed.top_level] == ["rm", "find", "echo"]


def test_pipelines_share_a_number() -> None:
    a, b, c = parse_command("curl x | sudo sh; ls").commands
    assert a.pipeline == b.pipeline != c.pipeline
    assert b.connector == "|"
    assert b.name == "sh"


@pytest.mark.parametrize(
    ("line", "writes"),
    [
        ("ls > out.txt", True),
        ("ls >> out.txt", True),
        ("ls &> out.txt", True),
        ("ls 2>/dev/null", False),
        ("ls 2>&1", False),
        ("ls >&-", False),
        ("cat < in.txt", False),
    ],
)
def test_which_redirections_write(line: str, writes: bool) -> None:
    (command,) = parse_command(line).commands
    assert command.writes_files is writes
    assert command.redirects


def test_one_plain_command_is_single() -> None:
    assert parse_command("git status").is_single
    assert not parse_command("git status > x").is_single
    assert not parse_command("git status | head").is_single
    assert not parse_command("echo $(date)").is_single


@pytest.mark.parametrize("line", ['echo "open', "echo $(date", "echo `date", "diff <(ls x"])
def test_a_line_that_does_not_close_is_not_parsed(line: str) -> None:
    parsed = parse_command(line)
    assert not parsed.parsed
    assert not parsed.is_single


def test_nesting_has_a_floor() -> None:
    assert parse_command("eval " * 4 + "reboot").parsed
    assert not parse_command("eval " * 12 + "reboot").parsed


def test_text_is_the_command_as_written() -> None:
    (command,) = parse_command("sudo  rm   -rf x").commands
    assert command.text == "sudo rm -rf x"


@pytest.mark.parametrize(
    ("line", "hidden"),
    [
        ('echo $(printf "%s)" "(" ; reboot)', "reboot"),
        ("echo $(echo 'a)' ; (reboot))", "reboot"),
        ("echo $(echo \\) ; reboot)", "reboot"),
        ("find . -exec xargs rm \\;", "rm"),
        ("sudo -- reboot", "reboot"),
        ("cat <<EOF\n\\$(not) $(reboot)\nEOF", "reboot"),
    ],
)
def test_substitutions_close_where_the_shell_closes_them(line: str, hidden: str) -> None:
    assert hidden in _programs(line)


def test_a_heredoc_that_never_ends_swallows_the_rest() -> None:
    assert _programs("cat <<EOF\nreboot") == ["cat"]
    assert _programs("cat <<") == ["cat"]


def test_an_unclosed_substitution_in_a_heredoc_stops_the_scan() -> None:
    parsed = parse_command("cat <<EOF\n$(reboot\nEOF")
    assert "reboot" not in _programs(parsed.source)


def test_odd_shapes_do_not_raise() -> None:
    for line in [
        "",
        "   ",
        "sudo",
        "timeout",
        "xargs",
        "sh -c",
        "bash -x",
        "find . -exec",
        "$((1+2))",
    ]:
        parse_command(line)  # never raises
    assert parse_command("").commands == ()
    (only,) = parse_command("sudo").commands
    assert only.argv == ()
    assert only.name == ""
    assert parse_command("bash -x script.sh").commands[0].name == "bash"
    assert [c.name for c in parse_command("sh -c").commands] == ["sh"]


def test_a_backslash_newline_joins_lines() -> None:
    assert _programs("git \\\n  status") == ["git"]
    assert parse_command("git \\\n  status").commands[0].argv == ("git", "status")


def test_an_escaped_backtick_inside_a_substitution() -> None:
    assert "date" in _programs("echo `date \\` x`")
