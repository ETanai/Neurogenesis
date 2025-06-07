import pytest
import src.__init__ as mod

def test___init___exists():
    assert hasattr(mod, "__init__")

# TODO: add more tests for src.__init__
