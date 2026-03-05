"""
MCP Feedback & Learning Tools - Analytics, pattern learning, and RAG feedback.

Includes:
- User ratings and feedback
- Internal analytics for verification of outputs
- Failed operation tracking and learning
- Session pattern analysis for LLM tuning
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
feedback_category = ToolCategory()
feedback_category.name = "feedback"
feedback_category.description = "Feedback, learning, and analytics tools"

# Database path for persistent learning
DB_PATH = Path("mcp_data/feedback.db")


def _get_db():
    """Get database connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY, run_id TEXT, rating INTEGER, comment TEXT, 
        created_at TEXT, metadata TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY, decision_type TEXT, query TEXT, result TEXT,
        success INTEGER, duration_ms REAL, created_at TEXT, session_id TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS failures (
        id INTEGER PRIMARY KEY, operation TEXT, error TEXT, context TEXT,
        created_at TEXT, resolved INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY, pattern_type TEXT, pattern_data TEXT,
        frequency INTEGER DEFAULT 1, last_seen TEXT)""")
    return conn


class SubmitRatingTool(BaseTool):
    name = "submit_rating"
    description = "Rate an analysis result for quality feedback."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("tool_name", "string", "Tool that produced result", required=True, examples=["create_chart", "detect_anomalies"]),
            ToolParameter("rating", "number", "Rating 1-5", required=True, examples=[4]),
            ToolParameter("comment", "string", "Optional feedback", examples=["chart was helpful"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        conn.execute(
            "INSERT INTO ratings (run_id, rating, comment, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (arguments["tool_name"], arguments["rating"], arguments.get("comment"),
             datetime.utcnow().isoformat(), json.dumps({"session": session.session_id if session else None}))
        )
        conn.commit()
        return success_response({"recorded": True, "run_id": arguments["tool_name"]})


class GetRatingsTool(BaseTool):
    name = "get_ratings"
    description = "Get aggregated ratings and feedback."
    category = "feedback"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("run_id", "string", "Filter by run ID")]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        run_id = arguments.get("run_id")
        
        if run_id:
            rows = conn.execute("SELECT rating, comment FROM ratings WHERE run_id=?", (run_id,)).fetchall()
        else:
            rows = conn.execute("SELECT rating FROM ratings").fetchall()
        
        if not rows:
            return success_response({"average": None, "count": 0})
        
        ratings = [r[0] for r in rows]
        return success_response({
            "average": round(sum(ratings) / len(ratings), 2),
            "count": len(ratings),
            "distribution": {i: ratings.count(i) for i in range(1, 6)}
        })


class LogDecisionTool(BaseTool):
    name = "log_decision"
    description = "Log a decision for auditing and learning."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("decision", "string", "Decision made", required=True, examples=["chose bar chart over pie chart"]),
            ToolParameter("reasoning", "string", "Why this decision", required=True, examples=["data has many categories"]),
            ToolParameter("alternatives", "array", "Other options considered", items={"type": "string"}, examples=[["pie chart", "table"]]),
            ToolParameter("context", "object", "Decision context"),
            ToolParameter("success", "boolean", "Whether it succeeded", default=True),
            ToolParameter("duration_ms", "number", "Execution time"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        conn.execute(
            "INSERT INTO decisions (decision_type, query, result, success, duration_ms, created_at, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (arguments["decision"], arguments.get("reasoning"), json.dumps({"alternatives": arguments.get("alternatives"), "context": arguments.get("context")}),
             1 if arguments.get("success", True) else 0, arguments.get("duration_ms"),
             datetime.utcnow().isoformat(), session.session_id if session else None)
        )
        conn.commit()
        return success_response({"logged": True})


class LogFailureTool(BaseTool):
    name = "log_failure"
    description = "Log a failed operation for learning and debugging."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("tool_name", "string", "Tool that failed", required=True, examples=["forecast"]),
            ToolParameter("error", "string", "Error message", required=True, examples=["insufficient data points"]),
            ToolParameter("context", "object", "Failure context"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        conn.execute(
            "INSERT INTO failures (operation, error, context, created_at) VALUES (?, ?, ?, ?)",
            (arguments["operation"], arguments["error"], json.dumps(arguments.get("context")),
             datetime.utcnow().isoformat())
        )
        conn.commit()
        logger.warning(f"Logged failure: {arguments['operation']} - {arguments['error']}")
        return success_response({"logged": True})


class GetFailureAnalyticsTool(BaseTool):
    name = "get_failure_analytics"
    description = "Analyze failure patterns for learning."
    category = "feedback"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("limit", "number", "Max failures to return", default=50)]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        rows = conn.execute(
            "SELECT operation, error, COUNT(*) as count FROM failures GROUP BY operation, error ORDER BY count DESC LIMIT ?",
            (arguments.get("limit", 50),)
        ).fetchall()
        
        failures = [{"operation": r[0], "error": r[1], "count": r[2]} for r in rows]
        total = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        unresolved = conn.execute("SELECT COUNT(*) FROM failures WHERE resolved=0").fetchone()[0]
        
        return success_response({
            "top_failures": failures,
            "total_failures": total,
            "unresolved": unresolved,
        })


class RecordPatternTool(BaseTool):
    name = "record_pattern"
    description = "Record a user pattern for LLM tuning."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("pattern_type", "string", "Pattern type", required=True,
                         enum=["query_style", "data_preference", "viz_preference", "workflow_sequence"]),
            ToolParameter("pattern_data", "object", "Pattern details", required=True),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        pattern_json = json.dumps(arguments["pattern_data"])
        
        # Check if pattern exists
        existing = conn.execute(
            "SELECT id, frequency FROM patterns WHERE pattern_type=? AND pattern_data=?",
            (arguments["pattern_type"], pattern_json)
        ).fetchone()
        
        if existing:
            conn.execute(
                "UPDATE patterns SET frequency=frequency+1, last_seen=? WHERE id=?",
                (datetime.utcnow().isoformat(), existing[0])
            )
        else:
            conn.execute(
                "INSERT INTO patterns (pattern_type, pattern_data, last_seen) VALUES (?, ?, ?)",
                (arguments["pattern_type"], pattern_json, datetime.utcnow().isoformat())
            )
        conn.commit()
        return success_response({"recorded": True})


class GetPatternsTool(BaseTool):
    name = "get_patterns"
    description = "Get learned patterns for LLM context."
    category = "feedback"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("pattern_type", "string", "Filter by type"),
            ToolParameter("min_frequency", "number", "Minimum frequency", default=2),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        conn = _get_db()
        
        query = "SELECT pattern_type, pattern_data, frequency FROM patterns WHERE frequency >= ?"
        params = [arguments.get("min_frequency", 2)]
        
        if arguments.get("pattern_type"):
            query += " AND pattern_type = ?"
            params.append(arguments["pattern_type"])
        
        query += " ORDER BY frequency DESC LIMIT 100"
        rows = conn.execute(query, params).fetchall()
        
        patterns = [{"type": r[0], "data": json.loads(r[1]), "frequency": r[2]} for r in rows]
        return success_response({"patterns": patterns, "count": len(patterns)})


class GetSessionAnalyticsTool(BaseTool):
    name = "get_session_analytics"
    description = "Get analytics for session context and LLM tuning."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        # Analyze session patterns
        tool_usage = {}
        for call in session.tool_calls:
            tool = call.get("tool", "unknown")
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        success_rate = sum(1 for c in session.tool_calls if c.get("success")) / max(len(session.tool_calls), 1)
        
        return success_response({
            "session_id": session.session_id,
            "tool_usage": tool_usage,
            "total_calls": len(session.tool_calls),
            "success_rate": round(success_rate, 2),
            "datasets_loaded": len(session.datasets),
            "models_trained": len(session.models),
            "charts_created": len(session.charts),
        })


class GenerateLLMContextTool(BaseTool):
    name = "generate_llm_context"
    description = "Generate context for LLM based on learned patterns and session history."
    category = "feedback"
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        # Get patterns
        conn = _get_db()
        patterns = conn.execute(
            "SELECT pattern_type, pattern_data, frequency FROM patterns WHERE frequency >= 2 ORDER BY frequency DESC LIMIT 20"
        ).fetchall()
        
        # Get recent successful decisions
        decisions = conn.execute(
            "SELECT decision_type, query, result FROM decisions WHERE success=1 ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        
        context = {
            "learned_preferences": [
                {"type": p[0], "data": json.loads(p[1]), "confidence": min(p[2] * 0.1, 1.0)}
                for p in patterns
            ],
            "recent_successes": [
                {"type": d[0], "query": d[1]}
                for d in decisions
            ],
        }
        
        if session:
            context["session_datasets"] = list(session.datasets.keys())
            context["recent_messages"] = session.conversation_history[-5:]
        
        # Format as system prompt addition
        prompt_addition = "User preferences based on history:\n"
        for pref in context["learned_preferences"][:5]:
            prompt_addition += f"- {pref['type']}: {pref['data']}\n"
        
        context["prompt_addition"] = prompt_addition
        return success_response(context)


feedback_category.register(SubmitRatingTool())
feedback_category.register(GetRatingsTool())
feedback_category.register(LogDecisionTool())
feedback_category.register(LogFailureTool())
feedback_category.register(GetFailureAnalyticsTool())
feedback_category.register(RecordPatternTool())
feedback_category.register(GetPatternsTool())
feedback_category.register(GetSessionAnalyticsTool())
feedback_category.register(GenerateLLMContextTool())
register_category(feedback_category)
