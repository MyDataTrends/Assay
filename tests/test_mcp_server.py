"""
Unit tests for MCP Server core and tools.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd
import tempfile
import os


# ============================================================================
# Config Tests
# ============================================================================

class TestMCPConfig:
    def test_default_config(self):
        from mcp_server.config import MCPConfig
        config = MCPConfig()
        assert config.port == 8766
        assert config.host == "127.0.0.1"
        assert config.enable_stdio is True
        assert config.enable_http is True
    
    def test_tool_enabled(self):
        from mcp_server.config import MCPConfig
        config = MCPConfig()
        assert config.is_tool_enabled("any_tool") is True
        
        config.disabled_tools = {"blocked_tool"}
        assert config.is_tool_enabled("blocked_tool") is False
        assert config.is_tool_enabled("allowed_tool") is True


# ============================================================================
# Session Tests
# ============================================================================

class TestMCPSession:
    def test_session_creation(self):
        from mcp_server.session import MCPSession
        session = MCPSession()
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
    
    def test_add_dataset(self):
        from mcp_server.session import MCPSession
        session = MCPSession()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        session.add_dataset("test_ds", df)
        
        assert "test_ds" in session.datasets
        assert session.get_dataset("test_ds") is not None
        assert len(session.datasets["test_ds"]) == 3
    
    def test_session_expiration(self):
        from mcp_server.session import MCPSession
        from datetime import datetime, timedelta
        
        session = MCPSession()
        session.last_accessed = datetime.utcnow() - timedelta(hours=3)
        assert session.is_expired(3600) is True  # 1 hour timeout
        assert session.is_expired(14400) is False  # 4 hour timeout


class TestSessionManager:
    def test_create_session(self):
        from mcp_server.session import SessionManager
        manager = SessionManager(max_sessions=5)
        session = manager.create_session()
        assert session is not None
        assert manager.get_session_count() == 1
    
    def test_max_sessions_eviction(self):
        from mcp_server.session import SessionManager
        manager = SessionManager(max_sessions=2)
        s1 = manager.create_session()
        s2 = manager.create_session()
        s3 = manager.create_session()  # Should evict s1
        
        assert manager.get_session_count() == 2
        assert manager.get_session(s1.session_id) is None


# ============================================================================
# Connector Tool Tests
# ============================================================================

class TestConnectorTools:
    def test_detect_source_type_csv(self):
        from mcp_server.tools.connectors import detect_source_type
        result = detect_source_type("test.csv")
        assert result["provider"] == "csv"
        assert result["source_type"] == "file"
    
    def test_detect_source_type_sqlite(self):
        from mcp_server.tools.connectors import detect_source_type
        result = detect_source_type("sqlite:///mydb.db")
        assert result["provider"] == "sqlite"
        assert result["source_type"] == "database"
    
    def test_detect_source_type_s3(self):
        from mcp_server.tools.connectors import detect_source_type
        result = detect_source_type("s3://bucket/key.csv")
        assert result["provider"] == "s3"
        assert result["source_type"] == "cloud"
    
    def test_detect_source_type_rest(self):
        from mcp_server.tools.connectors import detect_source_type
        result = detect_source_type("https://api.example.com/data")
        assert result["provider"] == "rest"
        assert result["source_type"] == "api"

    @pytest.mark.asyncio
    async def test_connect_file_csv(self):
        from mcp_server.tools.connectors import ConnectFileTool
        from mcp_server.session import MCPSession
        
        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            path = f.name
        
        try:
            tool = ConnectFileTool()
            session = MCPSession()
            result = await tool.execute({"path": path}, session=session)
            
            assert result["success"] is True
            assert result["data"]["rows"] == 2
        finally:
            os.unlink(path)


# ============================================================================
# Semantic Tool Tests
# ============================================================================

class TestSemanticTools:
    @pytest.mark.asyncio
    async def test_semantic_analyze_datasets(self):
        from mcp_server.tools.semantic import SemanticAnalyzeDatasetsTool
        from mcp_server.session import MCPSession
        
        session = MCPSession()
        df1 = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        df2 = pd.DataFrame({"id": [2, 3], "value": [10, 20]})
        session.add_dataset("ds1", df1)
        session.add_dataset("ds2", df2)
        
        tool = SemanticAnalyzeDatasetsTool()
        result = await tool.execute({"dataset_ids": ["ds1", "ds2"]}, session=session)
        
        assert result["success"] is True
        assert "id" in result["data"]["common_columns"]

    @pytest.mark.asyncio
    async def test_semantic_merge(self):
        from mcp_server.tools.semantic import SemanticMergeTool
        from mcp_server.session import MCPSession
        
        session = MCPSession()
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df2 = pd.DataFrame({"id": [2, 3, 4], "value": [10, 20, 30]})
        session.add_dataset("left", df1)
        session.add_dataset("right", df2)
        
        tool = SemanticMergeTool()
        result = await tool.execute({
            "left_dataset": "left",
            "right_dataset": "right",
            "left_on": ["id"],
            "right_on": ["id"],
            "how": "inner",
            "save_as": "merged"
        }, session=session)
        
        assert result["success"] is True
        assert result["data"]["rows"] == 2  # inner join on id=2,3


# ============================================================================
# Decision Tool Tests
# ============================================================================

class TestDecisionTools:
    @pytest.mark.asyncio
    async def test_decide_analysis_type_visualization(self):
        from mcp_server.tools.decision import DecideAnalysisTypeTool
        
        tool = DecideAnalysisTypeTool()
        result = await tool.execute({"query": "show me a chart of sales"})
        
        assert result["success"] is True
        assert result["data"]["action"] == "visualization"

    @pytest.mark.asyncio
    async def test_decide_analysis_type_modeling(self):
        from mcp_server.tools.decision import DecideAnalysisTypeTool
        
        tool = DecideAnalysisTypeTool()
        result = await tool.execute({"query": "predict next month's revenue"})
        
        assert result["success"] is True
        assert result["data"]["action"] == "modeling"


# ============================================================================
# Analysis Tool Tests
# ============================================================================

class TestAnalysisTools:
    @pytest.mark.asyncio
    async def test_describe_dataset(self):
        from mcp_server.tools.analysis import DescribeDatasetTool
        from mcp_server.session import MCPSession
        
        session = MCPSession()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        session.add_dataset("test", df)
        
        tool = DescribeDatasetTool()
        result = await tool.execute({"dataset_id": "test"}, session=session)
        
        assert result["success"] is True
        assert result["data"]["shape"]["rows"] == 3
        assert result["data"]["shape"]["columns"] == 2

    @pytest.mark.asyncio
    async def test_detect_anomalies(self):
        from mcp_server.tools.analysis import DetectAnomaliesTool
        from mcp_server.session import MCPSession
        import numpy as np
        
        session = MCPSession()
        # Create data with outliers
        data = [1, 2, 3, 2, 1, 2, 100]  # 100 is an outlier
        df = pd.DataFrame({"values": data})
        session.add_dataset("test", df)
        
        tool = DetectAnomaliesTool()
        result = await tool.execute({"dataset_id": "test", "method": "iqr"}, session=session)
        
        assert result["success"] is True
        assert "values" in result["data"]["anomalies"]


# ============================================================================
# Server Tests
# ============================================================================

class TestMCPServer:
    def test_server_creation(self):
        from mcp_server.server import MCPServer
        server = MCPServer()
        assert server.name == "assay-mcp"
        assert len(server._tools) > 0  # Should have built-in tools

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        from mcp_server.server import MCPServer
        server = MCPServer()
        
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = await server.handle_request(request)
        
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "assay-mcp"

    @pytest.mark.asyncio
    async def test_tools_list(self):
        from mcp_server.server import MCPServer
        server = MCPServer()
        
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        response = await server.handle_request(request)
        
        assert response["id"] == 1
        assert "tools" in response["result"]


# ============================================================================
# Feedback Tool Tests
# ============================================================================

class TestFeedbackTools:
    @pytest.mark.asyncio
    async def test_submit_rating(self):
        from mcp_server.tools.feedback import SubmitRatingTool
        
        tool = SubmitRatingTool()
        result = await tool.execute({"run_id": "test123", "rating": 5})
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_record_pattern(self):
        from mcp_server.tools.feedback import RecordPatternTool
        
        tool = RecordPatternTool()
        result = await tool.execute({
            "pattern_type": "query_style",
            "pattern_data": {"prefers": "visualizations"}
        })
        
        assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
