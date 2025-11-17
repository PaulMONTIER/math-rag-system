"""
Context variables for Langfuse span propagation.

This module provides thread-safe context variables for passing Langfuse spans
from workflow nodes to LLM clients without modifying function signatures.

Usage:
    # In workflow node:
    set_current_langfuse_span(span)

    # In LLM client:
    span = get_current_langfuse_span()
"""

from contextvars import ContextVar
from typing import Optional, Any

# Context variable for the current Langfuse span
_current_langfuse_span: ContextVar[Optional[Any]] = ContextVar(
    'langfuse_span',
    default=None
)


def set_current_langfuse_span(span: Optional[Any]) -> None:
    """
    Set the current Langfuse span in the context.

    This should be called by workflow nodes before calling agents.

    Args:
        span: Langfuse span object or None

    Example:
        >>> span = trace.span(name="classify", input={...})
        >>> set_current_langfuse_span(span)
        >>> # Now all LLM calls in this context will nest under this span
    """
    _current_langfuse_span.set(span)


def get_current_langfuse_span() -> Optional[Any]:
    """
    Get the current Langfuse span from the context.

    This should be called by LLM clients to get the parent span.

    Returns:
        Current Langfuse span or None if not set

    Example:
        >>> span = get_current_langfuse_span()
        >>> if span:
        ...     generation = span.generation(name="llm_call", ...)
    """
    return _current_langfuse_span.get()


def clear_langfuse_span() -> None:
    """
    Clear the current Langfuse span from the context.

    Optional cleanup, not strictly necessary as context vars are scoped.

    Example:
        >>> clear_langfuse_span()
    """
    _current_langfuse_span.set(None)
