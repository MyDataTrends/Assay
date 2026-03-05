"""
MCP LLM Management Tools - Control local model operations.
"""

import logging
from typing import Any, Dict, List
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
llm_category = ToolCategory()
llm_category.name = "llm"
llm_category.description = "Local LLM management tools"


class LLMStatusTool(BaseTool):
    name = "llm_status"
    description = "Get LLM server status and currently loaded model."
    category = "llm"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            manager = get_llm_subprocess()
            status = manager.get_status()
            return success_response(status)
        except ImportError:
            return error_response("LLM manager not available")


class LLMListModelsTool(BaseTool):
    name = "llm_list_models"
    description = "List available local GGUF models."
    category = "llm"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.scanner import scan_for_models
            models = scan_for_models()
            return success_response({"models": models, "count": len(models)})
        except ImportError:
            return error_response("LLM scanner not available")


class LLMLoadTool(BaseTool):
    name = "llm_load"
    description = "Load a specific GGUF model."
    category = "llm"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("model_path", "string", "Path to GGUF model file", required=True, examples=["models/llama-3-8b.gguf"]),
            ToolParameter("n_ctx", "number", "Context window size", default=2048),
            ToolParameter("n_gpu_layers", "number", "GPU layers to offload", default=0),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            manager = get_llm_subprocess()
            success = manager.load_model(
                arguments["model_path"],
                n_ctx=arguments.get("n_ctx", 2048),
                n_gpu_layers=arguments.get("n_gpu_layers", 0)
            )
            if success:
                return success_response({"loaded": True, "model": arguments["model_path"]})
            return error_response("Failed to load model")
        except ImportError:
            return error_response("LLM manager not available")


class LLMUnloadTool(BaseTool):
    name = "llm_unload"
    description = "Unload the current model."
    category = "llm"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return []
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            manager = get_llm_subprocess()
            manager.unload_model()
            return success_response({"unloaded": True})
        except ImportError:
            return error_response("LLM manager not available")


class LLMChatTool(BaseTool):
    name = "llm_chat"
    description = "Send chat completion request to local LLM."
    category = "llm"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("messages", "array", "Chat messages", required=True, items={
                "type": "object",
                "properties": {"role": {"type": "string"}, "content": {"type": "string"}}
            }, examples=[[{"role": "user", "content": "hello"}]]),
            ToolParameter("max_tokens", "number", "Max tokens to generate", default=256),
            ToolParameter("temperature", "number", "Sampling temperature", default=0.7),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            manager = get_llm_subprocess()
            
            response = manager.chat(
                arguments["messages"],
                max_tokens=arguments.get("max_tokens", 256),
                temperature=arguments.get("temperature", 0.7)
            )
            
            if session:
                for msg in arguments["messages"]:
                    session.add_message(msg["role"], msg["content"])
                session.add_message("assistant", response)
            
            return success_response({"response": response})
        except ImportError:
            return error_response("LLM manager not available")


class LLMCompleteTool(BaseTool):
    name = "llm_complete"
    description = "Send text completion request to local LLM."
    category = "llm"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("prompt", "string", "Prompt text", required=True, examples=["Once upon a time"]),
            ToolParameter("max_tokens", "number", "Max tokens", default=256),
            ToolParameter("temperature", "number", "Temperature", default=0.7),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            manager = get_llm_subprocess()
            
            response = manager.complete(
                arguments["prompt"],
                max_tokens=arguments.get("max_tokens", 256),
                temperature=arguments.get("temperature", 0.7)
            )
            
            return success_response({"completion": response})
        except ImportError:
            return error_response("LLM manager not available")


llm_category.register(LLMStatusTool())
llm_category.register(LLMListModelsTool())
llm_category.register(LLMLoadTool())
llm_category.register(LLMUnloadTool())
llm_category.register(LLMChatTool())
llm_category.register(LLMCompleteTool())
register_category(llm_category)
