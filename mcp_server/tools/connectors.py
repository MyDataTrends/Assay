"""
MCP Data Connector Tools.

Provides agentic data source discovery and connection capabilities.
The MCP can intelligently determine and connect to various data sources
without manual configuration.

Supported sources:
- Databases: SQLite, PostgreSQL, MySQL, SQL Server, MongoDB
- Files: CSV, Excel, Parquet, JSON, XML
- Cloud: S3, GCS, Azure Blob
- APIs: REST, GraphQL
"""
from __future__ import annotations  # Defer type annotation evaluation

import os
import re
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

# Lazy import for pandas - only imported when actually needed in tool execution
if TYPE_CHECKING:
    import pandas as pd

from . import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    register_category,
    success_response,
    error_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connection Registry
# =============================================================================

@dataclass
class ConnectionInfo:
    """Information about a data source connection."""
    connection_id: str
    source_type: str  # "database", "file", "cloud", "api"
    provider: str  # "sqlite", "postgresql", "csv", "s3", "rest", etc.
    connection_string: str
    metadata: Dict[str, Any]
    connection: Optional[Any] = None  # The actual connection object
    

# =============================================================================
# Data Source Detection
# =============================================================================

def detect_source_type(connection_string: str) -> Dict[str, Any]:
    """
    Analyze a connection string and detect the source type.
    
    Returns:
        Dict with source_type, provider, and parsed components
    """
    result = {
        "source_type": "unknown",
        "provider": "unknown",
        "components": {},
        "valid": False,
    }
    
    cs = connection_string.strip()
    
    # Check for file paths
    ext = Path(cs).suffix.lower()
    file_ext_map = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".xml": "xml",
        ".feather": "feather",
    }
    is_local_path = (
        os.path.exists(cs)
        or cs.startswith("./")
        or cs.startswith("../")
        or (ext in file_ext_map and "://" not in cs)
    )
    if is_local_path:
        result["source_type"] = "file"
        result["components"]["path"] = cs
        result["valid"] = True
        result["provider"] = file_ext_map.get(ext, "unknown")
        return result
    
    # Check for database connection strings
    db_patterns = [
        (r"^sqlite:///(.+)$", "database", "sqlite"),
        (r"^sqlite://(.+)$", "database", "sqlite"),
        (r"^postgresql://", "database", "postgresql"),
        (r"^postgres://", "database", "postgresql"),
        (r"^mysql://", "database", "mysql"),
        (r"^mysql\+pymysql://", "database", "mysql"),
        (r"^mssql://", "database", "sqlserver"),
        (r"^mssql\+pyodbc://", "database", "sqlserver"),
        (r"^mongodb://", "database", "mongodb"),
        (r"^mongodb\+srv://", "database", "mongodb"),
        (r"^redis://", "database", "redis"),
    ]
    
    for pattern, source_type, provider in db_patterns:
        if re.match(pattern, cs, re.IGNORECASE):
            result["source_type"] = source_type
            result["provider"] = provider
            result["valid"] = True
            
            # Parse URL components
            parsed = urlparse(cs)
            result["components"] = {
                "scheme": parsed.scheme,
                "host": parsed.hostname,
                "port": parsed.port,
                "database": parsed.path.lstrip("/"),
                "username": parsed.username,
            }
            return result
    
    # Check for cloud storage
    cloud_patterns = [
        (r"^s3://([^/]+)/(.*)$", "cloud", "s3"),
        (r"^gs://([^/]+)/(.*)$", "cloud", "gcs"),
        (r"^az://([^/]+)/(.*)$", "cloud", "azure"),
        (r"^azure://([^/]+)/(.*)$", "cloud", "azure"),
        (r"^https://([^.]+)\.s3\.amazonaws\.com/(.*)$", "cloud", "s3"),
        (r"^https://storage\.googleapis\.com/([^/]+)/(.*)$", "cloud", "gcs"),
        (r"^https://([^.]+)\.blob\.core\.windows\.net/(.*)$", "cloud", "azure"),
    ]
    
    for pattern, source_type, provider in cloud_patterns:
        match = re.match(pattern, cs, re.IGNORECASE)
        if match:
            result["source_type"] = source_type
            result["provider"] = provider
            result["valid"] = True
            result["components"] = {
                "bucket": match.group(1),
                "key": match.group(2) if len(match.groups()) > 1 else "",
            }
            return result
    
    # Check for API URLs
    if cs.startswith("http://") or cs.startswith("https://"):
        result["source_type"] = "api"
        result["valid"] = True
        
        parsed = urlparse(cs)
        result["components"] = {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "query": dict(parse_qs(parsed.query)),
        }
        
        # Try to detect API type
        if "/graphql" in cs.lower():
            result["provider"] = "graphql"
        elif "/odata" in cs.lower():
            result["provider"] = "odata"
        else:
            result["provider"] = "rest"
        
        return result
    
    return result


def scan_directory_for_data(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    Scan a directory for data files.
    
    Returns list of discovered data sources.
    """
    if extensions is None:
        extensions = [".csv", ".xlsx", ".xls", ".parquet", ".json", ".xml"]
    
    results = []
    path = Path(directory)
    
    if not path.exists():
        return results
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                stat = file_path.stat()
                results.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "provider": detect_source_type(str(file_path))["provider"],
                })
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")
    
    return results


def scan_env_for_connections() -> List[Dict[str, Any]]:
    """
    Scan environment variables for database connection strings.
    
    Looks for common patterns like DATABASE_URL, DB_CONNECTION, etc.
    """
    results = []
    
    patterns = [
        "DATABASE_URL",
        "DB_URL",
        "DB_CONNECTION",
        "POSTGRES_URL",
        "POSTGRESQL_URL",
        "MYSQL_URL",
        "MONGODB_URL",
        "MONGO_URL",
        "REDIS_URL",
        "SQL_SERVER_URL",
        "SQLSERVER_URL",
    ]
    
    # Also check for *_DATABASE_URL patterns
    for key, value in os.environ.items():
        if any(p in key.upper() for p in ["DATABASE", "DB_URL", "DB_CONNECTION"]):
            detected = detect_source_type(value)
            if detected["valid"]:
                results.append({
                    "env_var": key,
                    "source_type": detected["source_type"],
                    "provider": detected["provider"],
                    "connection_string": value[:50] + "..." if len(value) > 50 else value,
                })
    
    return results


# =============================================================================
# Connector Implementations
# =============================================================================

class BaseConnector:
    """Base class for data source connectors."""
    
    provider: str = "base"
    source_type: str = "unknown"
    
    def connect(self, connection_string: str, **kwargs) -> Any:
        """Establish connection."""
        raise NotImplementedError
    
    def disconnect(self, connection: Any) -> None:
        """Close connection."""
        pass
    
    def test(self, connection: Any) -> bool:
        """Test if connection is working."""
        return True
    
    def infer_schema(self, connection: Any, target: str) -> Dict[str, Any]:
        """Infer schema from the data source."""
        raise NotImplementedError
    
    def fetch_data(
        self, 
        connection: Any, 
        query_or_path: str,
        limit: Optional[int] = None
    ) -> "pd.DataFrame":
        """Fetch data as DataFrame."""
        raise NotImplementedError


class FileConnector(BaseConnector):
    """Connector for file-based data sources."""
    
    source_type = "file"
    
    def connect(self, connection_string: str, **kwargs) -> Dict[str, Any]:
        """For files, 'connection' is just the path info."""
        path = Path(connection_string)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {connection_string}")
        
        return {
            "path": str(path.absolute()),
            "provider": self.provider,
            "size": path.stat().st_size,
        }
    
    def test(self, connection: Dict[str, Any]) -> bool:
        """Test if file exists and is readable."""
        return os.path.exists(connection["path"])


class CSVConnector(FileConnector):
    """Connector for CSV files."""
    
    provider = "csv"
    
    def fetch_data(
        self,
        connection: Dict[str, Any],
        query_or_path: str = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> "pd.DataFrame":
        """Load CSV file."""
        import pandas as pd
        path = connection["path"]
        read_kwargs = {
            "nrows": limit,
            **kwargs
        }
        return pd.read_csv(path, **{k: v for k, v in read_kwargs.items() if v is not None})
    
    def infer_schema(self, connection: Dict[str, Any], target: str = None) -> Dict[str, Any]:
        """Infer schema from CSV file."""
        df = self.fetch_data(connection, limit=100)
        return {
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "nullable": df[col].isna().any(),
                    "sample_values": df[col].dropna().head(3).tolist(),
                }
                for col in df.columns
            ],
            "row_count_sample": len(df),
        }


class ExcelConnector(FileConnector):
    """Connector for Excel files."""
    
    provider = "excel"
    
    def fetch_data(
        self,
        connection: Dict[str, Any],
        query_or_path: str = None,
        limit: Optional[int] = None,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> "pd.DataFrame":
        """Load Excel file."""
        import pandas as pd
        path = connection["path"]
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        if limit:
            df = df.head(limit)
        return df
    
    def infer_schema(self, connection: Dict[str, Any], target: str = None) -> Dict[str, Any]:
        """Infer schema from Excel file."""
        import pandas as pd
        path = connection["path"]
        xl = pd.ExcelFile(path)
        
        sheets = []
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, nrows=10)
            sheets.append({
                "name": sheet,
                "columns": [
                    {"name": col, "dtype": str(df[col].dtype)}
                    for col in df.columns
                ],
            })
        
        return {"sheets": sheets}


class ParquetConnector(FileConnector):
    """Connector for Parquet files."""
    
    provider = "parquet"
    
    def fetch_data(
        self,
        connection: Dict[str, Any],
        query_or_path: str = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load Parquet file."""
        import pandas as pd
        path = connection["path"]
        df = pd.read_parquet(path, **kwargs)
        if limit:
            df = df.head(limit)
        return df


class JSONConnector(FileConnector):
    """Connector for JSON files."""
    
    provider = "json"
    
    def fetch_data(
        self,
        connection: Dict[str, Any],
        query_or_path: str = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load JSON file."""
        import pandas as pd
        path = connection["path"]
        df = pd.read_json(path, **kwargs)
        if limit:
            df = df.head(limit)
        return df


class SQLiteConnector(BaseConnector):
    """Connector for SQLite databases."""
    
    source_type = "database"
    provider = "sqlite"
    
    def connect(self, connection_string: str, **kwargs) -> Any:
        """Connect to SQLite database."""
        import sqlite3
        
        # Parse connection string
        if connection_string.startswith("sqlite:///"):
            path = connection_string[10:]
        elif connection_string.startswith("sqlite://"):
            path = connection_string[9:]
        else:
            path = connection_string
        
        return sqlite3.connect(path)
    
    def disconnect(self, connection: Any) -> None:
        """Close SQLite connection."""
        connection.close()
    
    def test(self, connection: Any) -> bool:
        """Test SQLite connection."""
        try:
            connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def infer_schema(self, connection: Any, target: str = None) -> Dict[str, Any]:
        """Get SQLite schema."""
        cursor = connection.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        result = {"tables": []}
        
        for table in tables:
            if target and table != target:
                continue
            
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [
                {
                    "name": row[1],
                    "dtype": row[2],
                    "nullable": not row[3],
                    "primary_key": bool(row[5]),
                }
                for row in cursor.fetchall()
            ]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            result["tables"].append({
                "name": table,
                "columns": columns,
                "row_count": row_count,
            })
        
        return result
    
    def fetch_data(
        self,
        connection: Any,
        query_or_path: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Execute query and return results."""
        import pandas as pd
        query = query_or_path
        if limit and "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"
        
        return pd.read_sql_query(query, connection)


class RESTConnector(BaseConnector):
    """Connector for REST APIs."""
    
    source_type = "api"
    provider = "rest"
    
    def connect(self, connection_string: str, **kwargs) -> Dict[str, Any]:
        """Store API base URL and auth info."""
        return {
            "base_url": connection_string.rstrip("/"),
            "headers": kwargs.get("headers", {}),
            "auth": kwargs.get("auth"),
        }
    
    def test(self, connection: Dict[str, Any]) -> bool:
        """Test API connectivity."""
        try:
            import requests
            response = requests.get(
                connection["base_url"],
                headers=connection.get("headers", {}),
                timeout=10
            )
            return response.status_code < 500
        except Exception:
            return False
    
    def fetch_data(
        self,
        connection: Dict[str, Any],
        query_or_path: str,
        limit: Optional[int] = None,
        method: str = "GET",
        params: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from REST API."""
        import requests
        import pandas as pd
        
        # Handle internal auth param (for FRED etc)
        headers = connection.get("headers", {}).copy()
        if "X-Internal-Auth-Param" in headers:
            param_str = headers.pop("X-Internal-Auth-Param")
            key, val = param_str.split("=", 1)
            if params is None:
                params = {}
            params[key] = val


        # Guard: strip leading path segment that duplicates the base_url suffix
        # e.g. base="…/fred", path="fred/series/…" → "…/fred/series/…"
        from urllib.parse import urlparse
        base_path_tail = urlparse(connection['base_url']).path.rstrip('/').rsplit('/', 1)[-1]
        stripped_path = query_or_path.lstrip('/')
        if base_path_tail and stripped_path.startswith(base_path_tail + '/'):
            stripped_path = stripped_path[len(base_path_tail) + 1:]
            logger.info(f"Stripped duplicate prefix '{base_path_tail}/' from endpoint")
        url = f"{connection['base_url']}/{stripped_path}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=120
                )
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=params,
                    timeout=120
                )
        except AttributeError as e:
            # Catch 'list' object has no attribute 'keys'
            raise ValueError(f"Request failed (AttributeError). Params type: {type(params)}. Params: {params}. Error: {e}") from e
        
        response.raise_for_status()
        data = response.json()
        
        data = response.json()
        
        # Use recursive table finder to locate the actual data
        from preprocessing.data_cleaning import find_table_data
        
        table_data = find_table_data(data)
        
        if table_data:
            df = pd.DataFrame(table_data)
        else:
            # Fallback for flat lists or single dicts
            df = pd.json_normalize(data) if isinstance(data, (dict, list)) else pd.DataFrame([data])
            
        # Infer types (convert string numbers to numeric)
            
        # Infer types (convert string numbers to numeric)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Attempt to convert to numeric, coercing errors to NaN
                    # We check if it looks numeric first to avoid aggressive conversion of IDs/names logic if needed
                    # But pd.to_numeric with errors='ignore' is safer?
                    # errors='coerce' turns non-numbers to NaN. We should check if the result is mostly non-NaN.
                    # Simple approach: apply to_numeric with ignore.
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    pass
        
        if limit:
            df = df.head(limit)
        
        return df


# Connector registry
CONNECTORS: Dict[str, BaseConnector] = {
    "csv": CSVConnector(),
    "excel": ExcelConnector(),
    "parquet": ParquetConnector(),
    "json": JSONConnector(),
    "sqlite": SQLiteConnector(),
    "rest": RESTConnector(),
}


def get_connector(provider: str) -> Optional[BaseConnector]:
    """Get a connector by provider name."""
    return CONNECTORS.get(provider)


# =============================================================================
# MCP Tools
# =============================================================================

connector_category = ToolCategory()
connector_category.name = "connectors"
connector_category.description = "Agentic data source discovery and connection tools"


class DiscoverDataSourcesTool(BaseTool):
    """Scan for available data sources in the environment."""
    
    name = "discover_data_sources"
    description = (
        "Scan the environment for available data sources including local files, "
        "database connections in environment variables, and configured cloud storage. "
        "Use this to find what data is available before connecting."
    )
    category = "connectors"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "scan_paths", "array", 
                "List of directories to scan for data files",
                items={"type": "string"},
            ),
            ToolParameter(
                "scan_env", "boolean",
                "Whether to scan environment variables for database URLs",
                default=True
            ),
            ToolParameter(
                "file_extensions", "array",
                "File extensions to look for (default: csv, xlsx, parquet, json)",
                items={"type": "string"},
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            arguments: Dictionary of arguments for the tool.
            session: The current session object.
        """
        results = {
            "files": [],
            "env_connections": [],
            "total_sources": 0,
        }
        
        # Scan directories
        scan_paths = arguments.get("scan_paths", ["."])
        extensions = arguments.get("file_extensions")
        
        for path in scan_paths:
            try:
                files = scan_directory_for_data(path, extensions)
                results["files"].extend(files)
            except Exception as e:
                logger.warning(f"Error scanning {path}: {e}")
        
        # Scan environment
        if arguments.get("scan_env", True):
            results["env_connections"] = scan_env_for_connections()
        
        results["total_sources"] = len(results["files"]) + len(results["env_connections"])
        
        return success_response(results, f"Found {results['total_sources']} data sources")


class AnalyzeConnectionStringTool(BaseTool):
    """Parse and validate a connection string."""
    
    name = "analyze_connection_string"
    description = (
        "Parse a connection string or path to determine the source type, "
        "provider, and connection parameters. Use this to understand how "
        "to connect to a data source."
    )
    category = "connectors"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_string", "string",
                "The connection string or file path to analyze",
                required=True
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        cs = arguments["connection_string"]
        result = detect_source_type(cs)
        
        if not result["valid"]:
            return error_response(f"Could not parse connection string: {cs}")
        
        # Add connector availability
        result["connector_available"] = result["provider"] in CONNECTORS
        
        return success_response(result)


class SuggestConnectionTool(BaseTool):
    """AI-powered connection suggestion."""
    
    name = "suggest_connection"
    description = (
        "Given a description of what data you need, suggest the best way to "
        "connect to it based on available data sources. Uses AI to match "
        "your request to discovered sources."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "description", "string",
                "Description of the data you need (e.g., 'sales data', 'customer records')",
                required=True
            ),
            ToolParameter(
                "context", "object",
                "Additional context (available sources, preferences)",
                properties={
                    "prefer_format": {"type": "string"},
                    "available_sources": {"type": "array"},
                }
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        description = arguments["description"].lower()
        context = arguments.get("context", {})
        
        # First, discover available sources
        files = scan_directory_for_data(".", recursive=True)
        env_connections = scan_env_for_connections()
        
        # Simple keyword matching (could be enhanced with LLM)
        suggestions = []
        
        keywords = description.split()
        
        for file_info in files:
            score = 0
            name_lower = file_info["name"].lower()
            
            for keyword in keywords:
                if keyword in name_lower:
                    score += 10
            
            if score > 0:
                suggestions.append({
                    "type": "file",
                    "path": file_info["path"],
                    "provider": file_info["provider"],
                    "score": score,
                    "reason": f"Filename contains matching keywords",
                })
        
        for env_conn in env_connections:
            score = 0
            var_lower = env_conn["env_var"].lower()
            
            for keyword in keywords:
                if keyword in var_lower:
                    score += 10
            
            if score > 0:
                suggestions.append({
                    "type": "database",
                    "env_var": env_conn["env_var"],
                    "provider": env_conn["provider"],
                    "score": score,
                    "reason": f"Environment variable matches keywords",
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        if not suggestions:
            return success_response({
                "suggestions": [],
                "message": "No matching data sources found. Try running discover_data_sources first."
            })
        
        return success_response({
            "suggestions": suggestions[:5],
            "best_match": suggestions[0],
        })


class ConnectFileTool(BaseTool):
    """Connect to a file-based data source."""
    
    name = "connect_file"
    description = (
        "Load data from a file (CSV, Excel, Parquet, JSON). "
        "Returns a connection ID that can be used for subsequent queries."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "path", "string",
                "Path to the file to load",
                required=True
            ),
            ToolParameter(
                "options", "object",
                "Optional loading options (encoding, sheet_name for Excel, etc.)",
                properties={
                    "encoding": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "delimiter": {"type": "string"},
                }
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        path = arguments["path"]
        options = arguments.get("options", {})
        
        # Detect file type
        detected = detect_source_type(path)
        if not detected["valid"] or detected["source_type"] != "file":
            return error_response(f"Invalid file path: {path}")
        
        provider = detected["provider"]
        connector = get_connector(provider)
        
        if not connector:
            return error_response(f"No connector available for {provider} files")
        
        try:
            connection = connector.connect(path, **options)
            connection_id = f"file_{uuid.uuid4().hex[:8]}"
            
            # Load data into session
            df = connector.fetch_data(connection, path)
            
            if session:
                session.add_dataset(connection_id, df, {
                    "source": path,
                    "provider": provider,
                })
                session.add_connection(connection_id, connection, {
                    "type": "file",
                    "provider": provider,
                    "path": path,
                })
            
            return success_response({
                "connection_id": connection_id,
                "provider": provider,
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            })
            
        except Exception as e:
            return error_response(f"Failed to load file: {e}")


class ConnectDatabaseTool(BaseTool):
    """Connect to a database."""
    
    name = "connect_database"
    description = (
        "Connect to a database using a connection string. "
        "Returns a connection ID for executing queries."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_string", "string",
                "Database connection string (e.g., sqlite:///data.db, postgresql://...)",
                required=True
            ),
            ToolParameter(
                "db_type", "string",
                "Database type (auto-detected if not specified)",
                enum=["sqlite", "postgresql", "mysql", "sqlserver", "mongodb"]
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        connection_string = arguments["connection_string"]
        db_type = arguments.get("db_type")
        
        # Detect database type
        detected = detect_source_type(connection_string)
        if not detected["valid"] or detected["source_type"] != "database":
            return error_response(f"Invalid database connection string")
        
        provider = db_type or detected["provider"]
        connector = get_connector(provider)
        
        if not connector:
            return error_response(f"No connector available for {provider}")
        
        try:
            connection = connector.connect(connection_string)
            connection_id = f"db_{uuid.uuid4().hex[:8]}"
            
            # Test connection
            if not connector.test(connection):
                return error_response("Connection test failed")
            
            if session:
                session.add_connection(connection_id, connection, {
                    "type": "database",
                    "provider": provider,
                })
            
            return success_response({
                "connection_id": connection_id,
                "provider": provider,
                "status": "connected",
            })
            
        except Exception as e:
            return error_response(f"Failed to connect: {e}")


class ConnectAPITool(BaseTool):
    """Connect to a REST API."""
    
    name = "connect_api"
    description = (
        "Connect to a REST or GraphQL API. "
        "Returns a connection ID for making requests."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "url", "string",
                "Base URL of the API (required if preset not used)",
                required=False,
                examples=["https://api.stlouisfed.org/fred", "https://api.worldbank.org/v2"]
            ),
            ToolParameter(
                "preset", "string",
                "Name of a pre-configured API preset (e.g., 'world_bank', 'fred')",
                required=False,
                examples=["fred", "world_bank", "kaggle"]
            ),
            ToolParameter(
                "auth", "object",
                "Authentication configuration",
                properties={
                    "type": {"type": "string", "enum": ["bearer", "api_key", "basic"]},
                    "token": {"type": "string"},
                    "header_name": {"type": "string"},
                }
            ),
            ToolParameter(
                "headers", "object",
                "Additional headers to include",
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        url = arguments.get("url")
        preset_name = arguments.get("preset")
        auth = arguments.get("auth", {})
        headers = arguments.get("headers", {})
        
        # Handle presets
        if preset_name:
            try:
                from config.api_presets import API_PRESETS
                if preset_name in API_PRESETS:
                    preset = API_PRESETS[preset_name]
                    url = preset["base_url"]
                    
                    # Handle auth from preset if not overridden
                    if not auth and "auth" in preset:
                        p_auth = preset["auth"]
                        auth_type = p_auth.get("type")
                        
                        if auth_type == "api_key":
                            env_var = p_auth.get("env_var")
                            if env_var and os.getenv(env_var):
                                auth = {
                                    "type": "api_key",
                                    "token": os.getenv(env_var),
                                    "header_name": p_auth.get("param_name") # or header_name logic
                                }
                                # Special handling for query param based auth (like FRED)
                                if p_auth.get("param_name") == "api_key":
                                      # This is tricky for generic REST connector. 
                                      # Usually we pass headers. 
                                      # If it needs query params, we might need to append to URL or handle in fetch.
                                      # For now, let's assume standard header auth or simple appending.
                                      pass
            except ImportError:
                pass
        
        if not url:
             return error_response("URL or valid preset name is required")

        # Add auth headers
        if auth:
            auth_type = auth.get("type", "bearer")
            token = auth.get("token", "")
            
            if auth_type == "bearer":
                headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "api_key":
                header_name = auth.get("header_name", "X-API-Key")
                # Check if this looks like a query param key (no dashes, lowercase, like 'api_key')
                # This is a bit hacky but works for FRED's 'api_key' vs standard 'X-Auth-Token'
                if "_" in header_name and "-" not in header_name:
                     # Query param auth (e.g. FRED api_key)
                     # Store in headers with special prefix for internal use
                     headers["X-Internal-Auth-Param"] = f"{header_name}={token}"
                else:
                    headers[header_name] = token
        
        connector = get_connector("rest")
        
        try:
            connection = connector.connect(url, headers=headers)
            connection_id = f"api_{uuid.uuid4().hex[:8]}"
            
            # Test connection
            if not connector.test(connection):
                logger.warning("API connection test returned non-2xx, but continuing")
            
            if session:
                session.add_connection(connection_id, connection, {
                    "type": "api",
                    "provider": "rest",
                    "base_url": url,
                    "preset": preset_name,
                    "auth_config": auth # Store auth config for re-use if needed (redacted?)
                })
            
            return success_response({
                "connection_id": connection_id,
                "base_url": url,
                "preset": preset_name,
                "status": "connected",
            })
            
        except Exception as e:
            return error_response(f"Failed to connect: {e}")


class ExecuteQueryTool(BaseTool):
    """Execute a SQL query on a connected database."""
    
    name = "execute_query"
    description = (
        "Execute a SQL query on a connected database and return results as a dataset."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_id", "string",
                "ID of the database connection",
                required=True
            ),
            ToolParameter(
                "query", "string",
                "SQL query to execute",
                required=True
            ),
            ToolParameter(
                "limit", "number",
                "Maximum rows to return",
                default=1000
            ),
            ToolParameter(
                "save_as", "string",
                "Dataset ID to save results as (optional)"
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        connection_id = arguments["connection_id"]
        query = arguments["query"]
        limit = arguments.get("limit", 1000)
        save_as = arguments.get("save_as")
        
        if not session:
            return error_response("Session required for query execution")
        
        connection = session.get_connection(connection_id)
        if not connection:
            return error_response(f"Connection not found: {connection_id}")
        
        # Get connector based on connection metadata
        conn_info = session.connections.get(connection_id, {})
        provider = conn_info.get("metadata", {}).get("provider", "sqlite")
        connector = get_connector(provider)
        
        if not connector:
            return error_response(f"No connector for provider: {provider}")
        
        try:
            df = connector.fetch_data(connection, query, limit=limit)
            
            # Optionally save as dataset
            if save_as:
                session.add_dataset(save_as, df, {
                    "source": "query",
                    "connection_id": connection_id,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                })
            
            # Return summary and sample
            return success_response({
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(10).to_dict(orient="records"),
                "dataset_id": save_as,
            })
            
        except Exception as e:
            return error_response(f"Query execution failed: {e}")


class FetchAPIDataTool(BaseTool):
    """Fetch data from a connected REST API."""
    
    name = "fetch_api_data"
    description = (
        "Fetch data from a connected REST API endpoint."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_id", "string",
                "ID of the API connection",
                required=True
            ),
            ToolParameter(
                "endpoint", "string",
                "API endpoint path (relative to base URL). Do NOT repeat the base URL prefix.",
                required=True,
                examples=["/series/observations?series_id=GDP&file_type=json", "/country/US/indicator/NY.GDP.MKTP.CD"]
            ),
            ToolParameter(
                "method", "string",
                "HTTP method",
                enum=["GET", "POST"],
                default="GET"
            ),
            ToolParameter(
                "params", "object",
                "Request parameters or body"
            ),
            ToolParameter(
                "save_as", "string",
                "Dataset ID to save results as"
            ),
            ToolParameter(
                "context", "string",
                "Semantic context (why this data is being fetched)"
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        connection_id = arguments["connection_id"]
        endpoint = arguments["endpoint"]
        method = arguments.get("method", "GET")
        params = arguments.get("params", {})
        save_as = arguments.get("save_as")
        context = arguments.get("context")
        
        if not session:
            return error_response("Session required")
        
        connection = session.get_connection(connection_id)
        if not connection:
            return error_response(f"Connection not found: {connection_id}")
        
        connector = get_connector("rest")
        
        try:
            df = connector.fetch_data(
                connection, 
                endpoint, 
                method=method,
                params=params
            )
            
            # Apply standardization pipeline
            from preprocessing.data_cleaning import standardize_dataframe
            df = standardize_dataframe(df)
            
            if save_as:
                # Retrieve description from preset if available
                description = "Fetched via API"
                try:
                    conn_info = session.connections.get(connection_id, {})
                    preset_name = conn_info.get("metadata", {}).get("preset")
                    if preset_name:
                         from config.api_presets import API_PRESETS
                         if preset_name in API_PRESETS:
                             description = API_PRESETS[preset_name].get("description", description)
                except Exception:
                    pass

                session.add_dataset(save_as, df, {
                    "source": "api",
                    "connection_id": connection_id,
                    "endpoint": endpoint,
                    "description": description,
                    "context": context  # The "why" and "what" from the caller
                })
            
            return success_response({
                "rows": len(df),
                "columns": list(df.columns),
                "sample": df.head(10).to_dict(orient="records"),
                "dataset_id": save_as,
            })
            
        except Exception as e:
            return error_response(f"API request failed: {e}")


class InferSchemaTool(BaseTool):
    """Automatically detect schema from a data source."""
    
    name = "infer_schema"
    description = (
        "Automatically detect the schema (columns, types, constraints) "
        "from a connected data source."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_id", "string",
                "ID of the connection",
                required=True
            ),
            ToolParameter(
                "target", "string",
                "Table name or specific target (optional, infer all if not specified)"
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        connection_id = arguments["connection_id"]
        target = arguments.get("target")
        
        if not session:
            return error_response("Session required")
        
        connection = session.get_connection(connection_id)
        if not connection:
            return error_response(f"Connection not found: {connection_id}")
        
        conn_info = session.connections.get(connection_id, {})
        provider = conn_info.get("metadata", {}).get("provider")
        connector = get_connector(provider)
        
        if not connector:
            return error_response(f"No connector for provider: {provider}")
        
        try:
            schema = connector.infer_schema(connection, target)
            return success_response(schema)
        except Exception as e:
            return error_response(f"Schema inference failed: {e}")


class ListConnectionsTool(BaseTool):
    """List all active data source connections."""
    
    name = "list_connections"
    description = "List all active data source connections in the current session."
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        connections = []
        for conn_id, conn_data in session.connections.items():
            connections.append({
                "connection_id": conn_id,
                "created_at": conn_data.get("created_at"),
                **conn_data.get("metadata", {})
            })
        
        return success_response({
            "connections": connections,
            "count": len(connections)
        })


class TestConnectionTool(BaseTool):
    """Validate that a connection is working."""
    
    name = "test_connection"
    description = "Test if a data source connection is still valid and responsive."
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "connection_id", "string",
                "ID of the connection to test",
                required=True
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        connection_id = arguments["connection_id"]
        
        if not session:
            return error_response("Session required")
        
        connection = session.get_connection(connection_id)
        if not connection:
            return error_response(f"Connection not found: {connection_id}")
        
        conn_info = session.connections.get(connection_id, {})
        provider = conn_info.get("metadata", {}).get("provider")
        connector = get_connector(provider)
        
        if not connector:
            return success_response({
                "connection_id": connection_id,
                "status": "unknown",
                "message": f"No connector available for {provider}"
            })
        
        try:
            is_valid = connector.test(connection)
            return success_response({
                "connection_id": connection_id,
                "status": "connected" if is_valid else "failed",
                "valid": is_valid,
            })
        except Exception as e:
            return success_response({
                "connection_id": connection_id,
                "status": "error",
                "valid": False,
                "error": str(e),
            })


class ProfileDataSourceTool(BaseTool):
    """Get statistics and quality metrics for a data source."""
    
    name = "profile_data_source"
    description = (
        "Analyze a connected data source to get statistics, data quality metrics, "
        "and recommendations."
    )
    category = "connectors"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "dataset_id", "string",
                "ID of the loaded dataset to profile",
                required=True
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        dataset_id = arguments["dataset_id"]
        
        if not session:
            return error_response("Session required")
        
        df = session.get_dataset(dataset_id)
        if df is None:
            return error_response(f"Dataset not found: {dataset_id}")
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "columns": [],
            "data_quality": {
                "total_missing": int(df.isna().sum().sum()),
                "complete_rows": int((~df.isna().any(axis=1)).sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            },
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "unique": int(df[col].nunique()),
                "unique_pct": round(df[col].nunique() / len(df) * 100, 2),
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info["stats"] = {
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                }
            else:
                top_values = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            profile["columns"].append(col_info)
        
        return success_response(profile)


# =============================================================================
# Meta-tools: suggest_tool
# =============================================================================

class SuggestToolTool(BaseTool):
    name = "suggest_tool"
    description = (
        "Find the best MCP tool for a task. Accepts a natural language "
        "description and returns the top matching tools with full schemas."
    )
    category = "meta"
    requires_session = False

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "query", "string",
                "Natural language description of what you want to do",
                required=True,
                examples=["plot GDP over time", "connect to FRED API", "run a forecast"],
            ),
        ]

    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        from mcp_server.tools import _tool_categories

        query = arguments["query"].lower()
        keywords = set(query.split())

        scored = []
        for cat in _tool_categories.values():
            for tool in cat.get_tools():
                text = f"{tool.name} {tool.description} {tool.category}".lower()
                overlap = len(keywords & set(text.split()))
                # Boost partial matches in name/description
                partial = sum(1 for kw in keywords if kw in text)
                score = overlap * 2 + partial
                if score > 0:
                    scored.append((score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]

        if not top:
            return success_response({
                "matches": [],
                "hint": "No matching tools found. Try broader keywords.",
            })

        matches = []
        for _score, tool in top:
            params = tool.get_parameters()
            matches.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "parameters": [
                    {"name": p.name, "type": p.type, "required": p.required,
                     "description": p.description}
                    for p in params
                ],
            })

        return success_response({"matches": matches})


# =============================================================================
# Convenience tools: fetch_fred_series
# =============================================================================

class FetchFredSeriesTool(BaseTool):
    name = "fetch_fred_series"
    description = (
        "Fetch a FRED economic data series in one call. "
        "Just provide a series ID (e.g. GDP, UNRATE, CPIAUCSL) and "
        "the tool handles API connection, endpoint construction, and "
        "dataset storage automatically."
    )
    category = "connectors"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "series_id", "string",
                "FRED series ID",
                required=True,
                examples=["GDP", "UNRATE", "CPIAUCSL", "PAYEMS"],
            ),
            ToolParameter("start_date", "string", "Start date (YYYY-MM-DD)", required=False),
            ToolParameter("end_date", "string", "End date (YYYY-MM-DD)", required=False),
        ]

    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import os, requests, pandas as pd

        series_id = arguments["series_id"].upper()
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return error_response(
                "FRED_API_KEY not set.  "
                "Get one at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set it as an environment variable."
            )

        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        if arguments.get("start_date"):
            params["observation_start"] = arguments["start_date"]
        if arguments.get("end_date"):
            params["observation_end"] = arguments["end_date"]

        url = "https://api.stlouisfed.org/fred/series/observations"
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return error_response(f"FRED API request failed: {e}")

        observations = data.get("observations", [])
        if not observations:
            return error_response(f"No observations returned for series '{series_id}'")

        df = pd.DataFrame(observations)
        # Clean up: convert date, make value numeric
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        dataset_id = f"fred_{series_id.lower()}"
        if session:
            session.add_dataset(dataset_id, df)

        return success_response({
            "dataset_id": dataset_id,
            "series_id": series_id,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df["date"].min()) if "date" in df.columns else None,
                "end": str(df["date"].max()) if "date" in df.columns else None,
            },
            "sample": df.head(3).to_dict(orient="records"),
        })


# Register all tools
connector_category.register(DiscoverDataSourcesTool())
connector_category.register(AnalyzeConnectionStringTool())
connector_category.register(SuggestConnectionTool())
connector_category.register(ConnectFileTool())
connector_category.register(ConnectDatabaseTool())
connector_category.register(ConnectAPITool())
connector_category.register(ExecuteQueryTool())
connector_category.register(FetchAPIDataTool())
connector_category.register(InferSchemaTool())
connector_category.register(ListConnectionsTool())
connector_category.register(TestConnectionTool())
connector_category.register(ProfileDataSourceTool())
connector_category.register(FetchFredSeriesTool())

# Meta-tools use their own category
meta_category = ToolCategory()
meta_category.name = "meta"
meta_category.description = "Tools for discovering and navigating available tools"
meta_category.register(SuggestToolTool())
register_category(meta_category)

register_category(connector_category)
