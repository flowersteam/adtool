from adtool.utils.depinj.injector import inject_callbacks
from adtool.utils.depinj.tests.testdeps import hw, logger
from adtool.utils.leaf.tests.test_leaf import DummyModule


def test_inject_callbacks(capsys):
    """NOTE: pytest is used to capture stderr"""
    a = DummyModule("hello")
    inject_callbacks(a, callbacks=[logger, hw])
    assert a.logger
    assert a.hw
    assert a.raise_callbacks

    a.hw()
    expected_out = "hello world\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    a.logger()
    expected_out = str(a.__dict__) + "\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    a.raise_callbacks()
    expected_out = str(a.__dict__) + "\nhello world\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    return


def test_inject_callbacks_args(capsys):
    a = DummyModule("hello")
    inject_callbacks(a, callbacks=[logger, hw])

    a.logger("arg0", kw0="kw0")
    expected_out = str(a.__dict__) + "\narg0\n" + "kw0, kw0\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out
