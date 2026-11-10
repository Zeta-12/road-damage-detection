import pytest

from road_damage.cli import main

COMMANDS = ["download", "prepare", "stats", "train", "evaluate", "duplicates"]


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as e:
        main([command, "--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_prepare_then_stats(rdd_root, capsys):
    args = ["--data-root", str(rdd_root), "--countries", "Japan", "Norway"]
    main(["prepare", *args])
    assert "| Japan | 120 |" in capsys.readouterr().out
    main(["stats", *args, "--sequences"])
    out = capsys.readouterr().out
    assert "| Norway | 60 |" in out
    assert "ratio" in out
