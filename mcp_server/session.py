"""
MCP Session Management.

Handles session state for multi-step MCP workflows, including
dataset storage, workflow tracking, and conversation history.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from threading import Lock

# Lazy import for pandas - only import when actually needed
if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MCPSession:
    """
    Represents a single MCP session with its state.
    
    Sessions track datasets, workflows, generated charts, and
    conversation history for multi-step analysis pipelines.
    """
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # User identification
    user_id: Optional[str] = None
    
    # Loaded datasets (dataset_id -> DataFrame)
    datasets: Dict[str, "pd.DataFrame"] = field(default_factory=dict)
    dataset_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Active data source connections
    connections: Dict[str, Any] = field(default_factory=dict)
    
    # Active workflows (workflow_id -> WorkflowManager)
    workflows: Dict[str, Any] = field(default_factory=dict)
    
    # Generated charts (chart_id -> chart data)
    charts: Dict[str, Any] = field(default_factory=dict)
    
    # Trained models (model_id -> model info)
    models: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation history for context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tool call history
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired."""
        expiry_time = self.last_accessed + timedelta(seconds=timeout_seconds)
        return datetime.utcnow() > expiry_time
    
    def add_dataset(
        self, 
        dataset_id: str, 
        df: "pd.DataFrame",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a dataset to the session."""
        self.datasets[dataset_id] = df
        self.dataset_metadata[dataset_id] = metadata or {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "added_at": datetime.utcnow().isoformat(),
        }
        self.touch()
    
    def get_dataset(self, dataset_id: str) -> Optional["pd.DataFrame"]:
        """Get a dataset by ID."""
        self.touch()
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in this session."""
        result = []
        for dataset_id, meta in self.dataset_metadata.items():
            result.append({
                "dataset_id": dataset_id,
                **meta
            })
        return result
    
    def add_connection(self, connection_id: str, connection: Any, metadata: Dict[str, Any]) -> None:
        """Add a data source connection."""
        self.connections[connection_id] = {
            "connection": connection,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.touch()
    
    def get_connection(self, connection_id: str) -> Optional[Any]:
        """Get a connection by ID."""
        self.touch()
        conn_info = self.connections.get(connection_id)
        return conn_info["connection"] if conn_info else None
    
    def add_chart(self, chart_id: str, chart_data: Any, metadata: Dict[str, Any]) -> None:
        """Add a generated chart."""
        self.charts[chart_id] = {
            "data": chart_data,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.touch()
    
    def add_model(self, model_id: str, model: Any, metadata: Dict[str, Any]) -> None:
        """Add a trained model."""
        self.models[model_id] = {
            "model": model,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.touch()
    
    def log_tool_call(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        result: Any,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log a tool call for auditing."""
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
            "result_summary": str(result)[:500] if result else None,
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.touch()
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.touch()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary (for status/inspection)."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "datasets": self.list_datasets(),
            "connections": [
                {"connection_id": k, **v["metadata"]} 
                for k, v in self.connections.items()
            ],
            "workflows": list(self.workflows.keys()),
            "charts": list(self.charts.keys()),
            "models": list(self.models.keys()),
            "tool_call_count": len(self.tool_calls),
            "message_count": len(self.conversation_history),
        }


class SessionManager:
    """
    Manages multiple MCP sessions with thread-safe access.
    
    Handles session creation, retrieval, expiration, and cleanup.
    """
    
    def __init__(self, max_sessions: int = 10, session_timeout: int = 3600):
        self._sessions: Dict[str, MCPSession] = {}
        self._lock = Lock()
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
    
    def create_session(self, user_id: Optional[str] = None) -> MCPSession:
        """Create a new session."""
        with self._lock:
            # Clean expired sessions first
            self._cleanup_expired()
            
            # Check capacity
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest session
                oldest = min(
                    self._sessions.values(),
                    key=lambda s: s.last_accessed
                )
                del self._sessions[oldest.session_id]
                logger.warning(f"Evicted session {oldest.session_id} due to capacity")
            
            session = MCPSession(user_id=user_id)
            self._sessions[session.session_id] = session
            logger.info(f"Created session {session.session_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[MCPSession]:
        """Get a session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired(self.session_timeout):
                del self._sessions[session_id]
                logger.info(f"Session {session_id} expired")
                return None
            if session:
                session.touch()
            return session
    
    def get_or_create_session(
        self, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> MCPSession:
        """Get existing session or create new one.
        
        When a session_id is supplied (e.g. "default" from single-user mode)
        but doesn't exist yet, we pin the new session to that exact ID so
        subsequent lookups with the same key will hit.
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
            # Create a session pinned to the requested ID
            with self._lock:
                self._cleanup_expired()
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(
                        self._sessions.values(),
                        key=lambda s: s.last_accessed,
                    )
                    del self._sessions[oldest.session_id]
                    logger.warning(f"Evicted session {oldest.session_id} due to capacity")
                session = MCPSession(session_id=session_id, user_id=user_id)
                self._sessions[session_id] = session
                logger.info(f"Created pinned session {session_id}")
                return session
        return self.create_session(user_id=user_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        with self._lock:
            self._cleanup_expired()
            return [s.to_dict() for s in self._sessions.values()]
    
    def _cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)
    
    def get_session_count(self) -> int:
        """Get count of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def get_last_session(self) -> Optional[MCPSession]:
        """Get the most recently accessed session."""
        with self._lock:
            if not self._sessions:
                return None
            return max(self._sessions.values(), key=lambda s: s.last_accessed)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        from .config import get_config
        config = get_config()
        _session_manager = SessionManager(
            max_sessions=config.max_concurrent_sessions,
            session_timeout=config.session_timeout_seconds
        )
    return _session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager."""
    global _session_manager
    _session_manager = manager
