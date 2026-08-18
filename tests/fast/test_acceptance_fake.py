"""Regression for retiring the old production Fake answer path."""

from langley.main import create_app
from langley.settings import Settings


def test_create_app_does_not_register_the_retired_fake_answer_path() -> None:
    app = create_app(Settings(environment="test"))

    assert not hasattr(app.state, "fake_answer")
