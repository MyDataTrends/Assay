"""
MCP API Discovery Tools.

Provides tools for:
- Searching the API registry based on user intent
- Managing API credentials securely
- Routing queries to appropriate data sources
"""
import os
import logging
from typing import Dict, List, Any, Optional

from . import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    register_category,
    success_response,
    error_response,
)

from mcp_server.api_registry import (
    API_REGISTRY,
    get_api,
    get_all_apis,
    search_apis_by_query,
    get_auth_instructions,
)
from mcp_server.credential_manager import (
    CredentialManager,
    get_or_prompt_credential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Category
# =============================================================================

api_discovery_category = ToolCategory()
api_discovery_category.name = "api_discovery"
api_discovery_category.description = "Tools for discovering and connecting to external data APIs"


# =============================================================================
# Tools
# =============================================================================

class SearchDataSourcesTool(BaseTool):
    """
    Search for APIs that can provide the data a user needs.
    
    This is the primary entry point for the autonomous data agent.
    Given a natural language description of what data is needed,
    it returns matching APIs ranked by relevance.
    """
    
    name = "search_data_sources"
    description = (
        "Search for external data APIs that can provide specific types of data. "
        "Give a natural language description of what data you need (e.g., "
        "'US unemployment trends', 'stock prices for AAPL', 'weather in London') "
        "and get back ranked API suggestions with setup instructions."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "query", "string",
                "Natural language description of the data you need",
                required=True,
            ),
            ToolParameter(
                "limit", "integer",
                "Maximum number of results to return",
                default=5,
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        if not query:
            return error_response("Query is required")
        
        # Use semantic (embedding-based) search with LLM fallback
        try:
            from mcp_server.semantic_router import semantic_search_apis
            api_matches = semantic_search_apis(query, top_k=limit)
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to keywords: {e}")
            # Fallback to keyword search
            keyword_matches = search_apis_by_query(query)
            api_matches = []
            for m in keyword_matches[:limit]:
                api = get_api(m["api_id"])
                if api:
                    from mcp_server.semantic_router import APIMatch
                    api_matches.append(APIMatch(
                        api_id=m["api_id"],
                        name=api.name,
                        description=api.description,
                        score=m["score"] / 50.0,
                        confidence="medium",
                        matched_via="keyword",
                    ))
        
        if not api_matches:
            return success_response({
                "matches": [],
                "suggestion": (
                    "No matching APIs found. Try describing your data needs differently, "
                    "or check available APIs with list_available_apis."
                ),
            })
        
        # Enrich with credential status
        cred_mgr = CredentialManager()
        results = []
        for match in api_matches:
            api = get_api(match.api_id)
            if api and api.auth_type != "none":
                env_var = api.auth_config.get("env_var", "")
                has_env = bool(os.environ.get(env_var))
                has_stored = cred_mgr.has_credential(match.api_id)
                cred_status = "ready" if (has_env or has_stored) else "needs_setup"
            else:
                cred_status = "not_required"
            
            results.append({
                "api_id": match.api_id,
                "name": match.name,
                "description": match.description,
                "score": round(match.score, 4),
                "confidence": match.confidence,
                "matched_via": match.matched_via,
                "credentials_status": cred_status,
            })
        
        return success_response({
            "matches": results,
            "total_found": len(results),
            "query": query,
            "search_method": "semantic" if api_matches and api_matches[0].matched_via != "keyword" else "keyword",
        })


class ListAvailableAPIsTool(BaseTool):
    """List all APIs available in the registry."""
    
    name = "list_available_apis"
    description = (
        "List all external data APIs available in the registry. "
        "Shows name, description, and whether authentication is required."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "category", "string",
                "Filter by category (economic, government, weather, news, health, geographic)",
                required=False,
            ),
            ToolParameter(
                "auth_required", "boolean",
                "Filter by authentication requirement",
                required=False,
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        apis = get_all_apis()
        
        results = []
        for api in apis:
            if not api.enabled:
                continue
            
            results.append({
                "id": api.id,
                "name": api.name,
                "description": api.description,
                "auth_required": api.auth_type != "none",
                "free_tier": api.free_tier,
                "data_types": api.data_types[:5],  # Limit for readability
                "signup_url": api.signup_url,
            })
        
        # Apply filters
        auth_filter = arguments.get("auth_required")
        if auth_filter is not None:
            results = [r for r in results if r["auth_required"] == auth_filter]
        
        return success_response({
            "apis": results,
            "total": len(results),
        })


class GetAPIAuthInfoTool(BaseTool):
    """Get authentication instructions for a specific API."""
    
    name = "get_api_auth_info"
    description = (
        "Get detailed authentication instructions for a specific API. "
        "Returns signup URL, environment variable name, and step-by-step instructions."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "api_id", "string",
                "ID of the API to get auth info for (e.g., 'fred', 'alpha_vantage')",
                required=True,
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        api_id = arguments.get("api_id", "")
        
        info = get_auth_instructions(api_id)
        
        if not info:
            return error_response(f"API '{api_id}' not found in registry")
        
        return success_response(info)


class StoreAPICredentialTool(BaseTool):
    """Securely store an API credential."""
    
    name = "store_api_credential"
    description = (
        "Securely store an API key using encrypted storage. "
        "The key is encrypted with AES-128 using your master password."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "api_id", "string",
                "ID of the API (e.g., 'fred', 'alpha_vantage')",
                required=True,
            ),
            ToolParameter(
                "api_key", "string",
                "The API key to store",
                required=True,
            ),
            ToolParameter(
                "master_password", "string",
                "Password to encrypt the credential (remember this!)",
                required=True,
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        api_id = arguments.get("api_id", "")
        api_key = arguments.get("api_key", "")
        master_password = arguments.get("master_password", "")
        
        if not all([api_id, api_key, master_password]):
            return error_response("api_id, api_key, and master_password are all required")
        
        # Validate API exists
        api = get_api(api_id)
        if not api:
            return error_response(f"API '{api_id}' not found in registry")
        
        try:
            cred_mgr = CredentialManager()
            cred_mgr.store_credential(api_id, api_key, master_password)
            
            return success_response({
                "stored": True,
                "api_id": api_id,
                "api_name": api.name,
                "message": f"API key for {api.name} stored securely.",
            })
        except Exception as e:
            return error_response(f"Failed to store credential: {e}")


class CheckAPICredentialsTool(BaseTool):
    """Check which APIs have credentials configured."""
    
    name = "check_api_credentials"
    description = (
        "Check which APIs have credentials configured (either via environment "
        "variables or encrypted storage). Helps identify what's ready to use."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        cred_mgr = CredentialManager()
        stored_creds = {c["api_id"] for c in cred_mgr.list_credentials()}
        
        results = []
        
        for api in get_all_apis():
            if api.auth_type == "none":
                results.append({
                    "api_id": api.id,
                    "name": api.name,
                    "status": "no_auth_required",
                    "ready": True,
                })
            else:
                env_var = api.auth_config.get("env_var", "")
                has_env = bool(os.environ.get(env_var))
                has_stored = api.id in stored_creds
                
                if has_env:
                    status = "env_var_set"
                elif has_stored:
                    status = "stored_encrypted"
                else:
                    status = "not_configured"
                
                results.append({
                    "api_id": api.id,
                    "name": api.name,
                    "status": status,
                    "ready": has_env or has_stored,
                    "env_var": env_var,
                    "signup_url": api.signup_url,
                })
        
        ready_count = sum(1 for r in results if r["ready"])
        
        return success_response({
            "credentials": results,
            "summary": {
                "total_apis": len(results),
                "ready": ready_count,
                "needs_setup": len(results) - ready_count,
            },
        })


class RouteQueryToAPITool(BaseTool):
    """
    Automatically route a data query to the best matching API.
    
    This is the "magic" tool that combines intent matching with
    credential checking and returns a ready-to-use connection.
    """
    
    name = "route_query_to_api"
    description = (
        "Automatically find the best API for your data query, check credentials, "
        "and return connection details. This is the one-stop tool for getting data."
    )
    category = "api_discovery"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "query", "string",
                "Natural language description of data needed",
                required=True,
            ),
            ToolParameter(
                "master_password", "string",
                "Password for encrypted credentials (if needed)",
                required=False,
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        query = arguments.get("query", "")
        master_password = arguments.get("master_password")
        
        if not query:
            return error_response("Query is required")
        
        # Use semantic search to find matching APIs
        try:
            from mcp_server.semantic_router import semantic_search_apis, get_router
            api_matches = semantic_search_apis(query, top_k=5)
        except Exception as e:
            logger.warning(f"Semantic router failed: {e}")
            # Fallback to keyword search
            keyword_matches = search_apis_by_query(query)
            api_matches = []
            for m in keyword_matches[:5]:
                api = get_api(m["api_id"])
                if api:
                    from mcp_server.semantic_router import APIMatch
                    api_matches.append(APIMatch(
                        api_id=m["api_id"],
                        name=api.name,
                        description=api.description,
                        score=m["score"] / 50.0,
                        confidence="medium",
                        matched_via="keyword",
                    ))
        
        if not api_matches:
            # Future: trigger dynamic API discovery here
            return success_response({
                "routed": False,
                "reason": "no_matching_api",
                "suggestion": "No APIs found that match your query. Try rephrasing.",
                "discovery_hint": "In future versions, this will trigger dynamic API discovery.",
            })
        
        # Try each match until we find one with valid credentials
        cred_mgr = CredentialManager()
        
        for match in api_matches:
            api = get_api(match.api_id)
            if not api:
                continue
            
            # Check if auth is needed and available
            if api.auth_type == "none":
                return success_response({
                    "routed": True,
                    "api_id": api.id,
                    "api_name": api.name,
                    "base_url": api.base_url,
                    "auth_headers": {},
                    "endpoints": [
                        {"path": e.path, "params": e.params}
                        for e in api.endpoints
                    ],
                    "suggestion": f"Use {api.name} - no authentication required!",
                })
            
            # Check for credentials
            env_var = api.auth_config.get("env_var", "")
            env_key = os.environ.get(env_var)
            
            if env_key:
                # Build auth based on location
                auth = self._build_auth(api, env_key)
                return success_response({
                    "routed": True,
                    "api_id": api.id,
                    "api_name": api.name,
                    "base_url": api.base_url,
                    "auth": auth,
                    "endpoints": [
                        {"path": e.path, "params": e.params}
                        for e in api.endpoints
                    ],
                    "credential_source": "environment",
                })
            
            # Try encrypted storage
            if cred_mgr.has_credential(api.id) and master_password:
                stored_key = cred_mgr.get_credential(api.id, master_password)
                if stored_key:
                    auth = self._build_auth(api, stored_key)
                    return success_response({
                        "routed": True,
                        "api_id": api.id,
                        "api_name": api.name,
                        "base_url": api.base_url,
                        "auth": auth,
                        "endpoints": [
                            {"path": e.path, "params": e.params}
                            for e in api.endpoints
                        ],
                        "credential_source": "encrypted_storage",
                    })
        
        # No API had valid credentials
        best_match = api_matches[0]
        api = get_api(best_match.api_id)
        
        return success_response({
            "routed": False,
            "reason": "credentials_required",
            "suggested_api": {
                "id": api.id,
                "name": api.name,
                "signup_url": api.signup_url,
                "env_var": api.auth_config.get("env_var"),
            },
            "instructions": get_auth_instructions(api.id)["instructions"],
        })
    
    def _build_auth(self, api, api_key: str) -> Dict[str, Any]:
        """Build auth dict based on API config."""
        location = api.auth_config.get("location", "query")
        
        if location == "query":
            param_name = api.auth_config.get("param_name", "api_key")
            return {"type": "query_param", "params": {param_name: api_key}}
        elif location == "header":
            header_name = api.auth_config.get("header_name", "Authorization")
            return {"type": "header", "headers": {header_name: api_key}}
        else:
            return {"type": "unknown", "api_key": api_key}


class GenerateDynamicConnectorTool(BaseTool):
    """
    Generate a connector for an API not in the registry.
    
    This tool enables autonomous discovery of new APIs by:
    1. Fetching and parsing API documentation
    2. Generating Python connector code
    3. Validating the generated code
    """
    
    name = "generate_dynamic_connector"
    description = """Generate a Python connector for a new API from its documentation URL.
    
    Use this when:
    - No existing API in the registry matches the user's data needs
    - User provides a URL to API documentation
    - User wants to connect to a custom or private API
    
    The tool will:
    1. Parse the API documentation (OpenAPI specs, HTML docs, or Swagger)
    2. Generate a Python connector with fetch_data method
    3. Validate the connector works correctly
    
    Returns the generated connector code and status."""
    
    parameters = [
        ToolParameter(
            name="docs_url",
            description="URL to the API documentation (OpenAPI spec, Swagger, or docs page)",
            type="string",
            required=True,
        ),
        ToolParameter(
            name="api_key",
            description="Optional API key to test the generated connector",
            type="string",
            required=False,
        ),
        ToolParameter(
            name="api_name",
            description="Optional name for the API (auto-detected if not provided)",
            type="string",
            required=False,
        ),
    ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        docs_url = arguments.get("docs_url", "").strip()
        api_key = arguments.get("api_key")
        api_name = arguments.get("api_name")
        
        if not docs_url:
            return error_response("docs_url is required")
        
        # Validate URL format
        if not docs_url.startswith(("http://", "https://")):
            return error_response("docs_url must be a valid HTTP/HTTPS URL")
        
        try:
            from mcp_server.dynamic_connector import (
                get_connector_manager,
                GeneratedConnector,
            )
            
            manager = get_connector_manager()
            result = manager.generate_connector(docs_url, api_key)
            
            if result.validated:
                return success_response({
                    "success": True,
                    "api_name": result.api_name,
                    "base_url": result.base_url,
                    "auth_type": result.auth_type,
                    "endpoints_found": len(result.endpoints),
                    "endpoints": result.endpoints[:5],  # First 5
                    "code_preview": result.code[:500] + "..." if len(result.code) > 500 else result.code,
                    "message": f"Successfully generated connector for {result.api_name}",
                    "usage": f"Use the generated connector with: connector = {result.api_name.replace(' ', '')}Connector(api_key='...')",
                })
            else:
                return success_response({
                    "success": False,
                    "api_name": result.api_name or "Unknown",
                    "error": result.error,
                    "partial_code": result.code[:300] if result.code else None,
                    "suggestion": "Try providing a direct link to an OpenAPI/Swagger specification for better results.",
                })
                
        except Exception as e:
            logger.error(f"Dynamic connector generation failed: {e}")
            return error_response(f"Generation failed: {str(e)}")


# =============================================================================
# Tool Registration
# =============================================================================

def get_api_discovery_tools() -> List[BaseTool]:
    """Get all API discovery tools."""
    return [
        SearchDataSourcesTool(),
        ListAvailableAPIsTool(),
        GetAPIAuthInfoTool(),
        StoreAPICredentialTool(),
        CheckAPICredentialsTool(),
        RouteQueryToAPITool(),
        GenerateDynamicConnectorTool(),
    ]

