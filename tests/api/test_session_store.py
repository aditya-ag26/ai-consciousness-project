import pytest

from src.api.session_store import Message, SessionStore


def test_created_session_starts_empty():
    store = SessionStore()

    session_id = store.create()

    assert store.exists(session_id)
    assert store.get_messages(session_id) == []


def test_messages_are_returned_in_order():
    store = SessionStore()
    session_id = store.create()

    store.append(session_id, Message(role="user", content="What is qualia?"))
    store.append(session_id, Message(role="assistant", content="Subjective experience."))

    assert [(m.role, m.content) for m in store.get_messages(session_id)] == [
        ("user", "What is qualia?"),
        ("assistant", "Subjective experience."),
    ]


def test_history_pairs_matches_the_shape_querybot_expects():
    store = SessionStore()
    session_id = store.create()
    store.append(session_id, Message(role="user", content="hi"))
    store.append(session_id, Message(role="assistant", content="hello"))

    assert store.history_pairs(session_id) == [("user", "hi"), ("assistant", "hello")]


def test_sessions_are_isolated_from_each_other():
    store = SessionStore()
    first, second = store.create(), store.create()

    store.append(first, Message(role="user", content="only in first"))

    assert store.get_messages(second) == []


def test_deleted_session_is_gone():
    store = SessionStore()
    session_id = store.create()

    store.delete(session_id)

    assert not store.exists(session_id)
    with pytest.raises(KeyError):
        store.get_messages(session_id)


def test_unknown_session_raises():
    store = SessionStore()

    with pytest.raises(KeyError):
        store.get_messages("does-not-exist")


def test_idle_sessions_expire():
    store = SessionStore(ttl_seconds=60)
    session_id = store.create()

    # Backdate the session past its TTL rather than sleeping through it.
    store._sessions[session_id].last_active -= 61

    assert not store.exists(session_id)


def test_active_sessions_are_not_expired():
    store = SessionStore(ttl_seconds=60)
    session_id = store.create()

    store._sessions[session_id].last_active -= 30

    assert store.exists(session_id)


def test_store_evicts_when_over_capacity():
    store = SessionStore(max_sessions=2)

    first = store.create()
    store.create()
    store.create()

    assert not store.exists(first)
