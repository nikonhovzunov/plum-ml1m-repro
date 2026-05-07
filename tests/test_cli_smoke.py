from plum_ml1m.cli import main


def test_cli_smoke_test(capsys):
    assert main(["smoke-test"]) == 0
    captured = capsys.readouterr()
    assert '"status": "ok"' in captured.out


def test_cli_validate_config_dir(capsys):
    assert main(["validate-config", "--config-dir", "configs"]) == 0
    captured = capsys.readouterr()
    assert "prepare_data.yaml" in captured.out
