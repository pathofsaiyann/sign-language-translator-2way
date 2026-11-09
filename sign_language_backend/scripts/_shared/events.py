"""Validated public events for /ws/sign-language; extras preserve CLI compatibility."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)


class GlossUpdate(Event):
    type: Literal['gloss_update'] = 'gloss_update'
    glosses: list[str]
    current_sign: str
    confidence: float = Field(ge=0, le=1)


class SentenceComplete(Event):
    type: Literal['sentence_complete'] = 'sentence_complete'
    sentence: str
    glosses: list[str]


class Status(Event):
    type: Literal['status'] = 'status'
    state: Literal['IDLE', 'SIGNING', 'SUBMITTING']


class Error(Event):
    type: Literal['error'] = 'error'
    error: str


def public_event(payload):
    """One event per frame, except SUBMITTING followed by sentence_complete."""
    value = dict(payload)
    if value.get('type'):
        # The two-way endpoint retains its typed staging/dispatch extension.
        return value
    status = value.get('status')
    if status == 'error':
        return Error(**value).model_dump()
    if 'sentence' in value:
        return SentenceComplete(**value).model_dump()
    if status == 'prediction':
        value['current_sign'] = value['prediction']
        return GlossUpdate(**value).model_dump()
    # Preserve internal gesture detail separately; public state has three values.
    internal = value.get('state')
    if internal:
        value['gesture_state'] = internal
    value['state'] = ('SUBMITTING' if status == 'submitting_sentence' else
                      'IDLE' if status == 'idle' else 'SIGNING')
    return Status(**value).model_dump()
