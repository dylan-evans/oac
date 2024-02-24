
import pytest

from .main import ChatSession, Mesg


@pytest.fixture
def history():
    return [
        Mesg("system", "You are an assistant"),
        Mesg("user", "What's an assistant?"),
        Mesg("assistant", "I am an assistant"),
        Mesg("user", "Why are you an assistant?"),
    ]


def test_filtering(history):
    sess = ChatSession()
    sess.trace.history = history

    user = list(sess.filter_messages("user"))
    assert len(user) == 2
    assert user[0][0] == 1 and user[0][1] == "What's an assistant?"
