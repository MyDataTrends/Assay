"""
MCP Workflow Tools - Multi-step pipeline orchestration.
"""

import logging
import uuid
from typing import Any, Dict, List
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
workflow_category = ToolCategory()
workflow_category.name = "workflow"
workflow_category.description = "Workflow orchestration tools"


class StartWorkflowTool(BaseTool):
    name = "start_workflow"
    description = "Initialize a new analysis workflow."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("user_id", "string", "User identifier", examples=["user_123"]),
            ToolParameter("file_name", "string", "Dataset file name", examples=["sales_data.csv"]),
            ToolParameter("description", "string", "Workflow description", examples=["Quarterly sales report"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        
        workflow = {
            "id": workflow_id,
            "user_id": arguments.get("user_id"),
            "file_name": arguments.get("file_name"),
            "description": arguments.get("description"),
            "steps": [],
            "status": "initialized",
        }
        
        if session:
            session.workflows[workflow_id] = workflow
        
        return success_response({"workflow_id": workflow_id, "status": "initialized"})


class WorkflowPreprocessTool(BaseTool):
    name = "workflow_preprocess"
    description = "Run preprocessing step on workflow data."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("workflow_id", "string", "Workflow ID", required=True, examples=["wf_abc123"]),
            ToolParameter("dataset_id", "string", "Dataset to preprocess", required=True, examples=["sales_2024"]),
            ToolParameter("config", "object", "Preprocessing config"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflow = session.workflows.get(arguments["workflow_id"])
        if not workflow:
            return error_response("Workflow not found")
        
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        config = arguments.get("config", {})
        original_rows = len(df)
        
        # Basic preprocessing
        if config.get("drop_duplicates", True):
            df = df.drop_duplicates()
        if config.get("drop_na_rows"):
            df = df.dropna()
        
        preprocessed_id = f"{arguments['dataset_id']}_preprocessed"
        session.add_dataset(preprocessed_id, df, {"source": "preprocessing"})
        
        workflow["steps"].append({"step": "preprocess", "input": arguments["dataset_id"], "output": preprocessed_id})
        workflow["status"] = "preprocessed"
        
        return success_response({
            "dataset_id": preprocessed_id,
            "original_rows": original_rows,
            "final_rows": len(df),
        })


class WorkflowEnrichTool(BaseTool):
    name = "workflow_enrich"
    description = "Run enrichment step on workflow data."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("workflow_id", "string", "Workflow ID", required=True, examples=["wf_abc123"]),
            ToolParameter("dataset_id", "string", "Dataset to enrich", required=True, examples=["sales_2024"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflow = session.workflows.get(arguments["workflow_id"])
        if not workflow:
            return error_response("Workflow not found")
        
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        try:
            from orchestration.semantic_enricher import SemanticEnricher
            enricher = SemanticEnricher()
            enriched_df = enricher.enrich(df)
            
            enriched_id = f"{arguments['dataset_id']}_enriched"
            session.add_dataset(enriched_id, enriched_df, {"source": "enrichment"})
            
            workflow["steps"].append({"step": "enrich", "input": arguments["dataset_id"], "output": enriched_id})
            workflow["status"] = "enriched"
            
            new_cols = [c for c in enriched_df.columns if c not in df.columns]
            return success_response({"dataset_id": enriched_id, "new_columns": new_cols})
        except ImportError:
            return success_response({"dataset_id": arguments["dataset_id"], "note": "Enrichment not available"})


class WorkflowAnalyzeTool(BaseTool):
    name = "workflow_analyze"
    description = "Run analysis step on workflow data."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("workflow_id", "string", "Workflow ID", required=True, examples=["wf_abc123"]),
            ToolParameter("dataset_id", "string", "Dataset to analyze", required=True, examples=["sales_2024"]),
            ToolParameter("intent", "object", "Analysis intent"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflow = session.workflows.get(arguments["workflow_id"])
        if not workflow:
            return error_response("Workflow not found")
        
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        intent = arguments.get("intent", {})
        
        try:
            from orchestration.analysis_selector import select_analyzer
            analyzer = select_analyzer(df)
            result = analyzer.run(df)
            
            workflow["steps"].append({"step": "analyze", "input": arguments["dataset_id"], "result": "completed"})
            workflow["status"] = "analyzed"
            
            return success_response({"result": result, "analyzer": type(analyzer).__name__})
        except ImportError:
            workflow["status"] = "analyzed"
            return success_response({"result": {"rows": len(df), "columns": list(df.columns)}, "note": "Basic analysis"})


class WorkflowGenerateOutputsTool(BaseTool):
    name = "workflow_generate_outputs"
    description = "Generate final workflow artifacts."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("workflow_id", "string", "Workflow ID", required=True, examples=["wf_abc123"])]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflow = session.workflows.get(arguments["workflow_id"])
        if not workflow:
            return error_response("Workflow not found")
        
        workflow["status"] = "completed"
        
        return success_response({
            "workflow_id": arguments["workflow_id"],
            "steps_completed": len(workflow["steps"]),
            "status": "completed",
        })


class GetWorkflowStatusTool(BaseTool):
    name = "get_workflow_status"
    description = "Check workflow progress."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("workflow_id", "string", "Workflow ID", required=True, examples=["wf_abc123"])]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflow = session.workflows.get(arguments["workflow_id"])
        if not workflow:
            return error_response("Workflow not found")
        
        return success_response(workflow)


class ListWorkflowsTool(BaseTool):
    name = "list_workflows"
    description = "List active and recent workflows."
    category = "workflow"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("user_id", "string", "Filter by user ID")]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        workflows = []
        user_id = arguments.get("user_id")
        
        for wf_id, wf in session.workflows.items():
            if user_id and wf.get("user_id") != user_id:
                continue
            workflows.append({"id": wf_id, "status": wf.get("status"), "steps": len(wf.get("steps", []))})
        
        return success_response({"workflows": workflows, "count": len(workflows)})


workflow_category.register(StartWorkflowTool())
workflow_category.register(WorkflowPreprocessTool())
workflow_category.register(WorkflowEnrichTool())
workflow_category.register(WorkflowAnalyzeTool())
workflow_category.register(WorkflowGenerateOutputsTool())
workflow_category.register(GetWorkflowStatusTool())
workflow_category.register(ListWorkflowsTool())
register_category(workflow_category)
