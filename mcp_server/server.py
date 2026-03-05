"""
MCP Server - Core Model Context Protocol implementation.

Provides JSON-RPC 2.0 based MCP server with support for:
- stdio transport (for Claude Desktop, MCP Inspector)
- HTTP transport (for API clients)
- Tool registration and discovery
- Resource endpoints
- Session management
"""

import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import time

from .config import get_config, MCPConfig
from .session import get_session_manager, MCPSession, SessionManager

logger = logging.getLogger(__name__)


# JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific error codes
TOOL_NOT_FOUND = -32001
TOOL_EXECUTION_ERROR = -32002
RESOURCE_NOT_FOUND = -32003
SESSION_ERROR = -32004
AUTH_ERROR = -32005


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    category: str = "general"
    requires_session: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "category": self.category,
        }


@dataclass
class ResourceDefinition:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class MCPServer:
    """
    Model Context Protocol Server.
    
    Handles JSON-RPC 2.0 messages over stdio or HTTP,
    manages tool registration, and coordinates with session manager.
    """
    
    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        session_manager: Optional[SessionManager] = None
    ):
        self.config = config or get_config()
        self.session_manager = session_manager or get_session_manager()
        
        # Registered tools and resources
        self._tools: Dict[str, ToolDefinition] = {}
        self._resources: Dict[str, ResourceDefinition] = {}
        
        # Server info
        self.name = "assay-mcp"
        self.version = "0.1.0"
        
        # Running state
        self._running = False
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register built-in server management tools."""
        # Session management
        self.register_tool(ToolDefinition(
            name="session_create",
            description="Create a new MCP session for multi-step analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier"
                    }
                }
            },
            handler=self._handle_session_create,
            category="session",
            requires_session=False
        ))
        
        self.register_tool(ToolDefinition(
            name="session_status",
            description="Get the status of the current session",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to check"
                    }
                },
                "required": ["session_id"]
            },
            handler=self._handle_session_status,
            category="session",
            requires_session=False
        ))
        
        self.register_tool(ToolDefinition(
            name="session_list",
            description="List all active sessions",
            input_schema={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_session_list,
            category="session",
            requires_session=False
        ))
    
    async def _handle_session_create(
        self, 
        arguments: Dict[str, Any], 
        session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
        """Handle session_create tool call."""
        user_id = arguments.get("user_id")
        session = self.session_manager.create_session(user_id=user_id)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "message": "Session created successfully"
        }
    
    async def _handle_session_status(
        self, 
        arguments: Dict[str, Any],
        session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
        """Handle session_status tool call."""
        session_id = arguments.get("session_id")
        target_session = self.session_manager.get_session(session_id)
        if not target_session:
            return {"error": f"Session {session_id} not found or expired"}
        return target_session.to_dict()
    
    async def _handle_session_list(
        self, 
        arguments: Dict[str, Any],
        session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
        """Handle session_list tool call."""
        sessions = self.session_manager.list_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
    
    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool with the server."""
        if not self.config.is_tool_enabled(tool.name):
            logger.debug(f"Tool {tool.name} disabled by configuration")
            return
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: ResourceDefinition) -> None:
        """Register a resource with the server."""
        self._resources[resource.uri] = resource
        logger.debug(f"Registered resource: {resource.uri}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of all registered tools."""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """Get list of all registered resources."""
        # Static resources
        resources = [res.to_dict() for res in self._resources.values()]
        
        # Dynamic resources from registry
        try:
            from .resources import list_all_resources
            # Get session from context if possible, or global manager
            # For listing, we might need a session context or list all possible
            # We'll retrieve global resources plus session-specific ones via providers
            providers_resources = list_all_resources()
            for res in providers_resources:
                resources.append({
                    "uri": res.uri,
                    "name": res.name,
                    "description": res.description,
                    "mimeType": res.mime_type
                })
        except ImportError:
            pass
            
        return resources
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a JSON-RPC 2.0 request.
        
        Args:
            request: JSON-RPC request object
            
        Returns:
            JSON-RPC response object
        """
        request_id = request.get("id")
        
        try:
            # Validate request
            if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
                return self._error_response(request_id, INVALID_REQUEST, "Invalid JSON-RPC version")
            
            method = request.get("method")
            params = request.get("params", {})
            
            if not method:
                return self._error_response(request_id, INVALID_REQUEST, "Missing method")
            
            # Route to appropriate handler
            result = await self._route_method(method, params)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {e}\n{traceback.format_exc()}")
            return self._error_response(request_id, INTERNAL_ERROR, str(e))
    
    async def _route_method(self, method: str, params: Dict[str, Any]) -> Any:
        """Route a method call to the appropriate handler."""
        
        # Initialize request
        if method == "initialize":
            return await self._handle_initialize(params)
        
        # Tool discovery
        if method == "tools/list":
            return {"tools": self.list_tools()}
        
        # Tool execution
        if method == "tools/call":
            return await self._handle_tool_call(params)
        
        # Resource discovery
        if method == "resources/list":
            return {"resources": self.list_resources()}
        
        # Resource read
        if method == "resources/read":
            return await self._handle_resource_read(params)
        
        # Ping
        if method == "ping":
            return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        
        raise ValueError(f"Unknown method: {method}")
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False, "listChanged": True},
            }
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        session_id = params.get("_meta", {}).get("session_id")
        
        # Single-user mode: default to a sticky "default" session
        if not session_id and self.config.single_user_mode:
            session_id = "default"
        
        if not tool_name:
            raise ValueError("Missing tool name")
        
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Get or create session
        session = None
        if tool.requires_session:
            session = self.session_manager.get_or_create_session(session_id)
        elif session_id:
            session = self.session_manager.get_session(session_id)
        
        # Execute tool
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(arguments, session=session)
            else:
                result = tool.handler(arguments, session=session)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log tool call
            if session and self.config.log_tool_calls:
                session.log_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration_ms,
                    success=True
                )
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, default=str)
                    }
                ],
                "_meta": {
                    "session_id": session.session_id if session else None,
                    "duration_ms": duration_ms
                }
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if session and self.config.log_tool_calls:
                session.log_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
            
            raise
    
    async def _handle_resource_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request."""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Missing resource URI")
        
        content = None
        mime_type = "application/json"
        
        # Check static resources
        if uri in self._resources:
            resource = self._resources[uri]
            mime_type = resource.mime_type
            if resource.handler:
                content = await resource.handler(uri)
            else:
                content = {"uri": uri, "available": True}
        else:
            # Check dynamic registry
            try:
                from .resources import read_resource
                # We need to pass the session context
                # The session_id might be in _meta from the request if it was passed
                # But for resources/read it's often a direct call.
                # We'll use the session manager to try to find a session if implied
                # For now, we'll pass None and let the provider handle it (or use default)
                # Ideally, we should get session_id from params
                # But MCP spec doesn't strictly define session handling for resources yet
                # We will assume global context or last active session if needed
                
                # Check directly
                content = await read_resource(uri, session=self.session_manager.get_last_session())
            except Exception as e:
                raise ValueError(f"Resource not found or error reading: {e}")
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": mime_type,
                    "text": json.dumps(content, default=str)
                }
            ]
        }
    
    def _error_response(
        self, 
        request_id: Any, 
        code: int, 
        message: str,
        data: Any = None
    ) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error
        }
    
    # =========================================================================
    # Stdio Transport
    # =========================================================================
    
    async def run_stdio(self) -> None:
        logger.info("Starting MCP server (stdio transport)")
        
        if sys.stdin.isatty():
            logger.error(
                "Stdio transport requires redirected pipes. "
                "Please run via an MCP client, or use the HTTP transport (--transport http)."
            )
            return

        self._running = True
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        
        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )
        
        while self._running:
            try:
                # Read line (JSON-RPC message)
                line = await reader.readline()
                if not line:
                    break
                
                # Parse request
                try:
                    request = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError as e:
                    response = self._error_response(None, PARSE_ERROR, str(e))
                    await self._write_response(writer, response)
                    continue
                
                # Handle request
                response = await self.handle_request(request)
                await self._write_response(writer, response)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stdio error: {e}")
                response = self._error_response(None, INTERNAL_ERROR, str(e))
                await self._write_response(writer, response)
        
        logger.info("MCP server (stdio) stopped")
    
    async def _write_response(
        self, 
        writer: asyncio.StreamWriter, 
        response: Dict[str, Any]
    ) -> None:
        """Write a JSON-RPC response."""
        data = json.dumps(response) + "\n"
        writer.write(data.encode('utf-8'))
        await writer.drain()
    
    # =========================================================================
    # HTTP Transport
    # =========================================================================
    
    async def run_http(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Run the MCP server using HTTP transport."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required for HTTP transport. Install with: pip install aiohttp")
            return
        
        host = host or self.config.host
        port = port or self.config.port
        
        app = web.Application()
        app.router.add_post("/mcp", self._http_handler)
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/tools", self._tools_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"MCP server (HTTP) running on http://{host}:{port}")
        self._running = True
        
        # Keep running
        while self._running:
            await asyncio.sleep(1)
        
        await runner.cleanup()
        logger.info("MCP server (HTTP) stopped")
    
    async def _http_handler(self, request) -> "web.Response":
        """Handle HTTP MCP requests."""
        from aiohttp import web
        
        # Check auth if required
        if self.config.require_auth:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
            else:
                api_key = request.headers.get("X-API-Key")
            
            if not self.config.validate_api_key(api_key):
                return web.json_response(
                    self._error_response(None, AUTH_ERROR, "Invalid API key"),
                    status=401
                )
        
        try:
            body = await request.json()
            response = await self.handle_request(body)
            return web.json_response(response)
        except json.JSONDecodeError as e:
            return web.json_response(
                self._error_response(None, PARSE_ERROR, str(e)),
                status=400
            )
    
    async def _health_handler(self, request) -> "web.Response":
        """Handle health check requests."""
        from aiohttp import web
        return web.json_response({
            "status": "healthy",
            "server": self.name,
            "version": self.version,
            "sessions": self.session_manager.get_session_count(),
            "tools": len(self._tools),
        })
    
    async def _tools_handler(self, request) -> "web.Response":
        """Handle tools listing requests."""
        from aiohttp import web
        grouped = {}
        for tool in self._tools.values():
            cat = tool.category or "general"
            grouped.setdefault(cat, []).append(tool.to_dict())
        return web.json_response({"tools": self.list_tools(), "by_category": grouped})
    
    def stop(self) -> None:
        """Stop the server."""
        self._running = False


def create_server(
    config: Optional[MCPConfig] = None,
    register_all_tools: bool = True
) -> MCPServer:
    """
    Create and configure an MCP server instance.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        register_all_tools: If True, register all tool categories.
        
    Returns:
        Configured MCPServer instance
    """
    server = MCPServer(config=config)
    
    if register_all_tools:
        # Import and register all tool categories
        try:
            from .tools import register_all_tools as register_tools
            register_tools(server)
        except ImportError:
            logger.warning("Tool modules not yet available")
            
        # Register resources
        try:
            from .resources import list_all_resources
            # Providers auto-register on import
            logger.info(f"Registered {len(list_all_resources())} resources")
        except ImportError:
            logger.warning("Resource modules not yet available")
    
    return server


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for running the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assay MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http", "both"],
        default="stdio",
        help="Transport mode"
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create server
    server = create_server()
    
    async def run():
        if args.transport == "stdio":
            await server.run_stdio()
        elif args.transport == "http":
            await server.run_http(args.host, args.port)
        else:
            # Run both
            await asyncio.gather(
                server.run_stdio(),
                server.run_http(args.host, args.port)
            )
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        server.stop()


import atexit
import signal

def _force_exit_handler():
    """Ensure all tasks are cancelled and loops stopped on exit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            for task in asyncio.all_tasks(loop):
                task.cancel()
            # Allow time for tasks to cancel
            # loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
            # loop.stop()
    except Exception:
        pass
    
    # Force kill any lingering threads/processes if needed
    # This addresses the 'Windows Error 6' and zombie processes
    # sys.exit(0) is standard, but os._exit(0) is the "hard" kill
    # We use os._exit(0) only as a last resort in the finally block of main()


if __name__ == "__main__":
    # Register the cleanup
    atexit.register(_force_exit_handler)
    
    # Also handle SIGINT/SIGTERM gracefully
    def _sig_handler(sig, frame):
        sys.exit(0)
    
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    
    main()
