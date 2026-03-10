"""
Unified Session Context Manager.

Provides a single source of truth for:
- Chat history (shared across all tabs)
- Dataset state (which datasets are loaded, which is active)
- LLM context (accumulated context from all user actions)
- Discovery state (API discovery results, pending fetches)

Usage:
    from ui.session_context import get_context
    
    ctx = get_context()
    ctx.add_message("user", "Show me sales trends")
    ctx.add_message("assistant", "Here's the analysis...")
    
    # Get chat history for any tab
    for msg in ctx.chat_history:
        print(f"{msg['role']}: {msg['content']}")
    
    # Get LLM context string
    prompt = ctx.build_llm_context() + "\n\nUser: " + user_input
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Session State Keys (Single source of truth)
# =============================================================================

# These are the canonical keys - all tabs should use these
KEYS = {
    "chat_history": "assay_chat_history",
    "datasets": "assay_datasets", 
    "primary_dataset": "assay_primary_dataset_id",
    "llm_context": "assay_llm_context",
    "action_log": "assay_action_log",
    "discoveries": "assay_discoveries",
    "pending_fetch": "assay_pending_fetch",
    "user_preferences": "assay_user_preferences",
}


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ActionLogEntry:
    """Log of a user action (for LLM context)."""
    action_type: str  # "upload", "analysis", "chart", "enrichment", "query"
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tab: str = ""  # Which tab the action occurred in
    result_summary: str = ""  # Brief summary of result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "description": self.description,
            "timestamp": self.timestamp,
            "tab": self.tab,
            "result_summary": self.result_summary,
        }


class SessionContext:
    """
    Unified session context manager.
    
    Wraps st.session_state with a clean API and ensures all tabs
    share the same data.
    """
    
    def __init__(self):
        """Initialize session state with defaults if needed."""
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """Ensure all session state keys exist with defaults."""
        if KEYS["chat_history"] not in st.session_state:
            st.session_state[KEYS["chat_history"]] = []
        
        if KEYS["datasets"] not in st.session_state:
            st.session_state[KEYS["datasets"]] = {}
        
        if KEYS["primary_dataset"] not in st.session_state:
            st.session_state[KEYS["primary_dataset"]] = None
        
        if KEYS["action_log"] not in st.session_state:
            st.session_state[KEYS["action_log"]] = []
        
        if KEYS["discoveries"] not in st.session_state:
            st.session_state[KEYS["discoveries"]] = []
        
        if KEYS["pending_fetch"] not in st.session_state:
            st.session_state[KEYS["pending_fetch"]] = None
        
        if KEYS["user_preferences"] not in st.session_state:
            st.session_state[KEYS["user_preferences"]] = {
                "auto_discovery": False,
                "show_confidence": True,
                "chart_style": "plotly",
            }
    
    # =========================================================================
    # Chat History
    # =========================================================================
    
    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """Get the shared chat history (as list of dicts for compatibility)."""
        # Ensure initialized before access
        if KEYS["chat_history"] not in st.session_state:
            st.session_state[KEYS["chat_history"]] = []
        return st.session_state[KEYS["chat_history"]]
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to chat history."""
        msg = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        st.session_state[KEYS["chat_history"]].append(msg.to_dict())
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent messages."""
        return self.chat_history[-n:]
    
    def clear_chat(self) -> None:
        """Clear chat history."""
        st.session_state[KEYS["chat_history"]] = []
    
    # =========================================================================
    # Datasets
    # =========================================================================
    
    @property
    def datasets(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded datasets."""
        return st.session_state[KEYS["datasets"]]
    
    @property
    def primary_dataset_id(self) -> Optional[str]:
        """Get the ID of the primary (active) dataset."""
        return st.session_state[KEYS["primary_dataset"]]
    
    @primary_dataset_id.setter
    def primary_dataset_id(self, value: str) -> None:
        """Set the primary dataset ID."""
        st.session_state[KEYS["primary_dataset"]] = value
    
    @property
    def primary_dataset(self) -> Optional[pd.DataFrame]:
        """Get the primary dataset DataFrame."""
        ds_id = self.primary_dataset_id
        if ds_id and ds_id in self.datasets:
            return self.datasets[ds_id]
        return None
    
    def add_dataset(
        self, 
        name: str, 
        df: pd.DataFrame, 
        set_as_primary: bool = False
    ) -> None:
        """Add a dataset."""
        st.session_state[KEYS["datasets"]][name] = df
        if set_as_primary or not self.primary_dataset_id:
            self.primary_dataset_id = name
        
        # Log the action
        self.log_action(
            "upload",
            f"Added dataset '{name}' with {len(df)} rows, {len(df.columns)} columns",
            result_summary=f"Columns: {', '.join(df.columns[:5])}..."
        )
    
    def remove_dataset(self, name: str) -> None:
        """Remove a dataset."""
        if name in self.datasets:
            del st.session_state[KEYS["datasets"]][name]
            if self.primary_dataset_id == name:
                remaining = list(self.datasets.keys())
                self.primary_dataset_id = remaining[0] if remaining else None
    
    # =========================================================================
    # Action Log (for LLM context)
    # =========================================================================
    
    @property
    def action_log(self) -> List[Dict[str, Any]]:
        """Get the action log."""
        return st.session_state[KEYS["action_log"]]
    
    def log_action(
        self,
        action_type: str,
        description: str,
        tab: str = "",
        result_summary: str = ""
    ) -> None:
        """Log a user action."""
        entry = ActionLogEntry(
            action_type=action_type,
            description=description,
            tab=tab,
            result_summary=result_summary,
        )
        st.session_state[KEYS["action_log"]].append(entry.to_dict())
        
        # Keep log size reasonable
        if len(st.session_state[KEYS["action_log"]]) > 100:
            st.session_state[KEYS["action_log"]] = st.session_state[KEYS["action_log"]][-50:]
    
    # =========================================================================
    # Discoveries (API discovery state)
    # =========================================================================
    
    @property
    def discoveries(self) -> List[Any]:
        """Get discovered APIs."""
        return st.session_state[KEYS["discoveries"]]
    
    @discoveries.setter
    def discoveries(self, value: List[Any]) -> None:
        st.session_state[KEYS["discoveries"]] = value
    
    @property
    def pending_fetch(self) -> Optional[Dict[str, Any]]:
        """Get pending API fetch."""
        return st.session_state[KEYS["pending_fetch"]]
    
    @pending_fetch.setter
    def pending_fetch(self, value: Optional[Dict[str, Any]]) -> None:
        st.session_state[KEYS["pending_fetch"]] = value
    
    # =========================================================================
    # User Preferences
    # =========================================================================
    
    @property
    def preferences(self) -> Dict[str, Any]:
        """Get user preferences."""
        return st.session_state[KEYS["user_preferences"]]
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        st.session_state[KEYS["user_preferences"]][key] = value
    
    # =========================================================================
    # LLM Context Building
    # =========================================================================
    
    def build_llm_context(
        self,
        include_chat: bool = True,
        include_datasets: bool = True,
        include_actions: bool = True,
        max_chat_messages: int = 10,
        max_actions: int = 10,
    ) -> str:
        """
        Build a comprehensive context string for LLM prompts.
        
        This provides the LLM with full awareness of:
        - Recent chat history
        - Loaded datasets and their structure
        - Recent user actions across all tabs
        """
        parts = []
        
        # Dataset context
        if include_datasets and self.datasets:
            ds_parts = ["## Current Data Context"]
            for name, df in self.datasets.items():
                is_primary = "(PRIMARY)" if name == self.primary_dataset_id else ""
                ds_parts.append(f"\n### Dataset: {name} {is_primary}")
                ds_parts.append(f"- Rows: {len(df)}, Columns: {len(df.columns)}")
                ds_parts.append(f"- Columns: {', '.join(df.columns.tolist())}")
                
                # Sample values
                if not df.empty:
                    sample = df.head(3).to_string(max_colwidth=30)
                    ds_parts.append(f"- Sample:\n```\n{sample}\n```")
            
            parts.append("\n".join(ds_parts))
        
        # Action context
        if include_actions and self.action_log:
            recent_actions = self.action_log[-max_actions:]
            action_parts = ["\n## Recent Actions"]
            for action in recent_actions:
                action_parts.append(f"- [{action['action_type']}] {action['description']}")
                if action.get('result_summary'):
                    action_parts.append(f"  → {action['result_summary']}")
            parts.append("\n".join(action_parts))
        
        # Chat context
        if include_chat and self.chat_history:
            recent_chat = self.get_recent_messages(max_chat_messages)
            chat_parts = ["\n## Recent Conversation"]
            for msg in recent_chat:
                role = msg['role'].upper()
                content = msg['content'][:500]  # Truncate long messages
                chat_parts.append(f"{role}: {content}")
            parts.append("\n".join(chat_parts))
        
        return "\n\n".join(parts)
    
    def get_dataset_summary(self) -> str:
        """Get a brief summary of loaded datasets for prompts."""
        if not self.datasets:
            return "No datasets loaded."
        
        summaries = []
        for name, df in self.datasets.items():
            is_primary = " (active)" if name == self.primary_dataset_id else ""
            summaries.append(f"- {name}{is_primary}: {len(df)} rows, columns: {', '.join(df.columns[:5])}")
        
        return "Loaded datasets:\n" + "\n".join(summaries)

    # =========================================================================
    # Session Lifecycle Management
    # =========================================================================
    
    def reset(self) -> None:
        """
        Reset session to fresh state.
        
        Clears all session data including:
        - Chat history
        - Loaded datasets
        - Action log
        - Discoveries
        
        Use this for a hard reset or at session end.
        """
        logger.info("Resetting session context")
        st.session_state[KEYS["chat_history"]] = []
        st.session_state[KEYS["datasets"]] = {}
        st.session_state[KEYS["primary_dataset"]] = None
        st.session_state[KEYS["action_log"]] = []
        st.session_state[KEYS["discoveries"]] = []
        st.session_state[KEYS["pending_fetch"]] = None
        # Note: preferences are preserved across reset
        
        # Clear caches
        self.clear_caches()
    
    def dispose(self) -> None:
        """
        Dispose of transient resources at session end.
        
        Unlike reset(), this:
        - Summarizes the session to InteractionLogger (if available)
        - Clears memory caches
        - Keeps preferences
        
        Call this when user navigates away or at session boundary.
        """
        logger.info("Disposing session resources")
        
        # Attempt to summarize before clearing
        try:
            from llm_learning.interaction_logger import get_interaction_logger
            logger_instance = get_interaction_logger()
            # Use a hash of first action timestamp as session ID
            if self.action_log:
                session_id = self.action_log[0].get("timestamp", "unknown")[:10]
                logger_instance.summarize_session(session_id)
        except Exception as e:
            logger.debug(f"Could not summarize session: {e}")
        
        # Clear caches but keep state for potential re-use
        self.clear_caches()
    
    def clear_caches(self) -> None:
        """Clear all memory caches (embeddings, etc)."""
        try:
            from learning.cache_utils import get_embedding_cache
            cache = get_embedding_cache()
            cache.clear()
            logger.debug("Cleared embedding cache")
        except Exception as e:
            logger.debug(f"Could not clear caches: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for this session."""
        stats = {
            "chat_messages": len(self.chat_history),
            "datasets_loaded": len(self.datasets),
            "action_log_entries": len(self.action_log),
            "discoveries": len(self.discoveries),
        }
        
        # Add dataset memory estimates
        total_rows = 0
        total_cols = 0
        for name, df in self.datasets.items():
            total_rows += len(df)
            total_cols += len(df.columns)
        stats["total_dataset_rows"] = total_rows
        stats["total_dataset_columns"] = total_cols
        
        # Add cache stats if available
        try:
            from learning.cache_utils import get_embedding_cache
            cache = get_embedding_cache()
            stats["embedding_cache"] = cache.get_stats()
        except Exception:
            pass
        
        return stats


# =============================================================================
# Global Access
# =============================================================================

_context: Optional[SessionContext] = None


def get_context() -> SessionContext:
    """Get the session context instance.
    Must evaluate per-run to ensure st.session_state is properly initialized.
    """
    return SessionContext()


# =============================================================================
# Migration Helpers (for gradual adoption)
# =============================================================================

def migrate_legacy_state():
    """
    Migrate old session state keys to new unified keys.
    
    Call this once at app startup to preserve existing data.
    """
    migrations = [
        # (old_key, new_key)
        ("chat_history", KEYS["chat_history"]),
        ("dashboard_chat_messages", KEYS["chat_history"]),
        ("datasets", KEYS["datasets"]),
        ("primary_dataset_id", KEYS["primary_dataset"]),
        ("discovered_apis", KEYS["discoveries"]),
        ("pending_api_fetch", KEYS["pending_fetch"]),
    ]
    
    for old_key, new_key in migrations:
        if old_key in st.session_state and old_key != new_key:
            # Only migrate if new key doesn't exist or is empty
            if new_key not in st.session_state or not st.session_state[new_key]:
                st.session_state[new_key] = st.session_state[old_key]
                logger.info(f"Migrated session state: {old_key} → {new_key}")


# =============================================================================
# Convenience Functions
# =============================================================================

def add_user_message(content: str) -> None:
    """Add a user message to chat history."""
    get_context().add_message("user", content)


def add_assistant_message(content: str) -> None:
    """Add an assistant message to chat history."""
    get_context().add_message("assistant", content)


def get_chat_for_display() -> List[Dict[str, str]]:
    """Get chat history formatted for st.chat_message display."""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in get_context().chat_history
    ]


def log_tab_action(tab_name: str, action: str, description: str = "") -> None:
    """Log an action with tab context."""
    get_context().log_action(
        action_type=action,
        description=description or action,
        tab=tab_name,
    )
