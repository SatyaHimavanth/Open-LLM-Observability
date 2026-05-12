import asyncio
import uuid
from universal_agent_obs.core import (
    _set_trace, _restore_trace, _current_trace, _current_span,
    _set_context, _restore_context, _current_project, _current_user, _current_tags, _current_metadata
)

async def test_trace_propagation():
    trace_id = str(uuid.uuid4())
    span_id = str(uuid.uuid4())

    assert _current_trace() is None

    tokens = _set_trace(trace_id, span_id)
    assert _current_trace() == trace_id
    assert _current_span() == span_id

    async def subtask():
        assert _current_trace() == trace_id
        assert _current_span() == span_id
        return True

    assert await asyncio.create_task(subtask())

    _restore_trace(tokens)
    assert _current_trace() is None
    assert _current_span() is None

async def test_context_propagation():
    project = "test_proj"
    user = {"id": "123"}
    tags = ["tag1"]
    meta = {"key": "val"}

    tokens = _set_context(project_name=project, user=user, tags=tags, metadata=meta)

    assert _current_project() == project
    assert _current_user() == user
    assert _current_tags() == tags
    assert _current_metadata() == meta

    async def subtask():
        assert _current_project() == project
        assert _current_user() == user
        assert _current_tags() == tags
        assert _current_metadata() == meta
        return True

    assert await asyncio.create_task(subtask())

    _restore_context(tokens)
    # default project name is "default"
    assert _current_project() == "default"
    assert _current_user() is None

if __name__ == "__main__":
    asyncio.run(test_trace_propagation())
    asyncio.run(test_context_propagation())
    print("Context propagation tests passed!")
