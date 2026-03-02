"""
MCP Server Configuration.

Handles configuration for the MCP server including transport settings,
authentication, tool enablement, and resource limits.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""
    
    # Server settings
    host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: get_int("MCP_PORT", 8766))
    
    # Transport modes
    enable_stdio: bool = field(default_factory=lambda: get_bool("MCP_ENABLE_STDIO", True))
    enable_http: bool = field(default_factory=lambda: get_bool("MCP_ENABLE_HTTP", True))
    
    # Single-user mode: when True, requests without a session_id are routed
    # to a persistent "default" session. Ideal for local single-user deployments.
    single_user_mode: bool = field(default_factory=lambda: get_bool("MCP_SINGLE_USER", True))
    
    # Authentication
    require_auth: bool = field(default_factory=lambda: get_bool("MCP_REQUIRE_AUTH", False))
    api_keys: Set[str] = field(default_factory=set)
    
    # Tool categories to enable (empty = all)
    enabled_tools: Set[str] = field(default_factory=set)
    disabled_tools: Set[str] = field(default_factory=set)
    
    # Resource limits
    max_concurrent_sessions: int = field(default_factory=lambda: get_int("MCP_MAX_SESSIONS", 10))
    session_timeout_seconds: int = field(default_factory=lambda: get_int("MCP_SESSION_TIMEOUT", 3600))
    max_dataset_rows: int = field(default_factory=lambda: get_int("MCP_MAX_DATASET_ROWS", 1000000))
    max_dataset_size_mb: int = field(default_factory=lambda: get_int("MCP_MAX_DATASET_SIZE_MB", 500))
    
    # Tool execution limits
    tool_timeout_seconds: int = field(default_factory=lambda: get_int("MCP_TOOL_TIMEOUT", 300))
    max_query_result_rows: int = field(default_factory=lambda: get_int("MCP_MAX_QUERY_ROWS", 10000))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("MCP_LOG_LEVEL", "INFO"))
    log_tool_calls: bool = field(default_factory=lambda: get_bool("MCP_LOG_TOOL_CALLS", True))
    
    # Data directories
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MCP_DATA_DIR", "mcp_data")))
    temp_dir: Path = field(default_factory=lambda: Path(os.getenv("MCP_TEMP_DIR", "mcp_temp")))
    
    # Self-orchestration (local LLM calling MCP tools)
    enable_self_orchestration: bool = field(
        default_factory=lambda: get_bool("MCP_ENABLE_SELF_ORCHESTRATION", True)
    )
    self_orchestration_max_iterations: int = field(
        default_factory=lambda: get_int("MCP_SELF_ORCHESTRATION_MAX_ITER", 10)
    )
    
    def __post_init__(self):
        """Initialize directories and load API keys."""
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API keys from environment
        api_keys_str = os.getenv("MCP_API_KEYS", "")
        if api_keys_str:
            self.api_keys = set(k.strip() for k in api_keys_str.split(",") if k.strip())
        
        # Load enabled/disabled tools from environment
        enabled_str = os.getenv("MCP_ENABLED_TOOLS", "")
        if enabled_str:
            self.enabled_tools = set(t.strip() for t in enabled_str.split(",") if t.strip())
        
        disabled_str = os.getenv("MCP_DISABLED_TOOLS", "")
        if disabled_str:
            self.disabled_tools = set(t.strip() for t in disabled_str.split(",") if t.strip())
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        if tool_name in self.disabled_tools:
            return False
        if self.enabled_tools and tool_name not in self.enabled_tools:
            return False
        return True
    
    def validate_api_key(self, key: Optional[str]) -> bool:
        """Validate an API key."""
        if not self.require_auth:
            return True
        if not key:
            return False
        return key in self.api_keys
    
    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MCPConfig":
        """Create configuration from a dictionary."""
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance


# Global config instance
_config: Optional[MCPConfig] = None


def get_config() -> MCPConfig:
    """Get the global MCP configuration."""
    global _config
    if _config is None:
        _config = MCPConfig.from_env()
    return _config


def set_config(config: MCPConfig) -> None:
    """Set the global MCP configuration."""
    global _config
    _config = config
