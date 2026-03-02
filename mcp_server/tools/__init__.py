"""
MCP Tool Registry and Base Classes.

Provides infrastructure for registering and managing MCP tools.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod
import logging
import functools

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types
    examples: Optional[List[Any]] = None  # Example values for this parameter


def build_input_schema(
    parameters: List[ToolParameter],
    additional_properties: bool = False
) -> Dict[str, Any]:
    """Build JSON Schema from parameter list."""
    properties = {}
    required = []
    
    for param in parameters:
        prop = {
            "type": param.type,
            "description": param.description,
        }
        if param.default is not None:
            prop["default"] = param.default
        if param.enum:
            prop["enum"] = param.enum
        if param.items:
            prop["items"] = param.items
        if param.properties:
            prop["properties"] = param.properties
        if param.examples:
            prop["examples"] = param.examples
        
        properties[param.name] = prop
        
        if param.required:
            required.append(param.name)
    
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    
    return schema


class BaseTool(ABC):
    """
    Abstract base class for MCP tools.
    
    Subclass this to create new tools with a consistent interface.
    """
    
    # Tool metadata - override in subclasses
    name: str = "base_tool"
    description: str = "Base tool description"
    category: str = "general"
    requires_session: bool = True
    
    def __init__(self):
        self._parameters: List[ToolParameter] = []
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Get the input schema for this tool."""
        return build_input_schema(self.get_parameters())
    
    def get_parameters(self) -> List[ToolParameter]:
        """Override to define tool parameters."""
        return self._parameters
    
    @abstractmethod
    async def execute(
        self,
        arguments: Dict[str, Any],
        session: Optional[Any] = None
    ) -> Any:
        """
        Execute the tool with given arguments.
        
        Args:
            arguments: Tool arguments matching the input schema
            session: Optional MCP session for state management
            
        Returns:
            Tool result (will be JSON serialized)
        """
        pass
    
    async def __call__(
        self,
        arguments: Dict[str, Any],
        session: Optional[Any] = None
    ) -> Any:
        """Allow tools to be called directly."""
        return await self.execute(arguments, session=session)
    
    def to_definition(self) -> "ToolDefinition":
        """Convert to a ToolDefinition for registration."""
        from ..server import ToolDefinition
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            handler=self.execute,
            category=self.category,
            requires_session=self.requires_session,
        )


class ToolCategory:
    """
    A collection of related tools.
    
    Use this to organize tools into categories that can be
    registered together with the MCP server.
    """
    
    name: str = "general"
    description: str = "General tools"
    
    def __init__(self):
        self._tools: List[BaseTool] = []
    
    def register(self, tool: BaseTool) -> BaseTool:
        """Register a tool with this category."""
        self._tools.append(tool)
        return tool
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools in this category."""
        return self._tools
    
    def register_all(self, server: "MCPServer") -> None:
        """Register all tools in this category with the server."""
        for tool in self._tools:
            server.register_tool(tool.to_definition())


# Tool decorator for simpler tool creation
def tool(
    name: str,
    description: str,
    parameters: List[ToolParameter],
    category: str = "general",
    requires_session: bool = True
):
    """
    Decorator to create a tool from a function.
    
    Usage:
        @tool(
            name="my_tool",
            description="Does something",
            parameters=[
                ToolParameter("arg1", "string", "First argument", required=True),
            ]
        )
        async def my_tool(arguments, session=None):
            return {"result": arguments["arg1"]}
    """
    def decorator(func: Callable) -> BaseTool:
        class DecoratedTool(BaseTool):
            pass
        
        DecoratedTool.name = name
        DecoratedTool.description = description
        DecoratedTool.category = category
        DecoratedTool.requires_session = requires_session
        
        tool_instance = DecoratedTool()
        tool_instance._parameters = parameters
        
        # Wrap the function as the execute method
        @functools.wraps(func)
        async def execute(self, arguments, session=None):
            return await func(arguments, session=session)
        
        tool_instance.execute = execute.__get__(tool_instance, DecoratedTool)
        
        return tool_instance
    
    return decorator


# =============================================================================
# Tool Registration
# =============================================================================

# Global registry of tool categories
_tool_categories: Dict[str, ToolCategory] = {}


def register_category(category: ToolCategory) -> ToolCategory:
    """Register a tool category globally."""
    _tool_categories[category.name] = category
    return category


def get_category(name: str) -> Optional[ToolCategory]:
    """Get a registered category by name."""
    return _tool_categories.get(name)


def list_categories() -> List[str]:
    """List all registered category names."""
    return list(_tool_categories.keys())


def register_all_tools(server: "MCPServer") -> int:
    """
    Register all tools from all categories with the server.
    
    Returns the total number of tools registered.
    """
    # Import all tool modules to trigger registration
    try:
        from . import (
            connectors,
            semantic,
            decision,
            visualization,
            analysis,
            workflow,
            llm,
            feedback,
            api_discovery,  # New: API discovery and credential management
        )
        
        # Manually register api_discovery tools (they use a different pattern)
        from .api_discovery import get_api_discovery_tools
        for tool in get_api_discovery_tools():
            server.register_tool(tool.to_definition())
            
    except ImportError as e:
        logger.warning(f"Some tool modules not available: {e}")
    
    total = 0
    for category in _tool_categories.values():
        category.register_all(server)
        total += len(category.get_tools())
    
    logger.info(f"Registered {total} tools from {len(_tool_categories)} categories")
    return total


# =============================================================================
# Common Response Helpers
# =============================================================================

def success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """Create a successful tool response."""
    result = {"success": True, "data": data}
    if message:
        result["message"] = message
    return result


def error_response(error: str, code: Optional[str] = None) -> Dict[str, Any]:
    """Create an error tool response."""
    result = {"success": False, "error": error}
    if code:
        result["code"] = code
    return result


def paginated_response(
    items: List[Any],
    total: int,
    offset: int = 0,
    limit: int = 100
) -> Dict[str, Any]:
    """Create a paginated response."""
    return {
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(items) < total
    }
