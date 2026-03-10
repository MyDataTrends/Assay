"""
Cascade Decision Planner for Assay.

Implements structured intent → plan → tool cascade:
- Deterministic rules for common cases
- LLM fallback for novel cases  
- Execution feedback loop (Generate-Check-Reflect)
- Learning from both successful paths

Usage:
    from orchestration.cascade_planner import CascadePlanner
    
    planner = CascadePlanner()
    plan = planner.plan(query, context={"df": df})
    result = planner.execute(plan)
"""
import logging
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Recognized user intents."""
    DESCRIBE_DATA = "describe_data"
    VISUALIZE = "visualize"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    MODEL_TRAIN = "model_train"
    MODEL_PREDICT = "model_predict"
    ENRICH_DATA = "enrich_data"
    EXPORT = "export"
    COMPARE = "compare"
    UNKNOWN = "unknown"


class PlanStatus(Enum):
    """Execution plan status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: str
    action: str
    tool: str
    inputs: Dict[str, Any]
    expected_output: str
    fallback_tool: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ExecutionPlan:
    """A complete execution plan."""
    plan_id: str
    intent: Intent
    query: str
    steps: List[PlanStep]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: PlanStatus = PlanStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent.value,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "status": self.status.value,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """Result of executing a plan."""
    plan_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 0
    execution_time_ms: int = 0
    learned: bool = False  # Whether this execution was logged for learning


# =============================================================================
# Intent Classification - Deterministic Rules
# =============================================================================

# Pattern-based intent classification (deterministic first)
INTENT_PATTERNS = {
    Intent.DESCRIBE_DATA: [
        r"\b(describe|profile|overview|summary|summarize|info|about|tell me about|show me|what is)\b.*\b(data|dataset|table|columns?|rows?)\b",
        r"\b(data|dataset)\b.*\b(look like|structure|shape|size)\b",
        r"\bhow many\b.*\b(rows?|columns?|records?)\b",
        r"\b(column|field)s?\b.*\b(types?|names?)\b",
    ],
    Intent.VISUALIZE: [
        r"\b(chart|graph|plot|visuali[sz]e|show|display|draw)\b",
        r"\b(bar|line|scatter|histogram|pie|heatmap|box)\b.*\b(chart|graph|plot)?\b",
        r"\b(trend|distribution|correlation)\b",
    ],
    Intent.FILTER: [
        r"\b(filter|where|only|exclude|remove|keep)\b",
        r"\b(greater|less|equal|between|contains?|starts?|ends?)\b",
        r"\brows? where\b",
    ],
    Intent.TRANSFORM: [
        r"\b(transform|clean|preprocess|prepare|modify|change|convert|rename|add|create)\b.*\b(column|data|field)\b",
        r"\b(fill|handle|impute)\b",
        r"\b(missing|null|na|nan)\b",
        r"\b(normalize|scale|encode|one.?hot)\b",
    ],
    Intent.AGGREGATE: [
        r"\b(group|aggregate|sum|count|average|avg|total|by)\b",
        r"\b(per|for each|breakdown)\b",
    ],
    Intent.MODEL_TRAIN: [
        r"\b(train|build|create|fit)\b.*\b(model|classifier|regressor|predictor)\b",
        r"\b(predict|forecast|classify)\b.*\b(using|with)\b.*\b(model)?\b",
        r"\b(machine learning|ml|regression|classification)\b",
    ],
    Intent.MODEL_PREDICT: [
        r"\b(predict|forecast|estimate|project)\b",
        r"\b(what will|what would|future|next)\b",
    ],
    Intent.ENRICH_DATA: [
        r"\b(enrich|augment|add|fetch|get|pull)\b.*\b(data|external|api)\b",
        r"\b(join|merge|combine)\b.*\b(with|from)\b.*\b(external|api|source)\b",
    ],
    Intent.EXPORT: [
        r"\b(export|save|download|generate)\b.*\b(csv|excel|pdf|report)\b",
        r"\b(create|make)\b.*\b(report|document)\b",
    ],
    Intent.COMPARE: [
        r"\b(compare|difference|vs|versus|between)\b",
        r"\b(which|what)\b.*\b(better|worse|higher|lower)\b",
    ],
}


def classify_intent(query: str) -> tuple[Intent, float]:
    """
    Classify user intent using deterministic rules.
    
    Returns:
        Tuple of (intent, confidence)
    """
    query_lower = query.lower().strip()
    
    best_intent = Intent.UNKNOWN
    best_score = 0.0
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Score based on pattern specificity
                specificity = len(pattern) / 100  # Longer patterns = more specific
                score = 0.7 + min(specificity, 0.25)  # Base 0.7, max 0.95
                
                if score > best_score:
                    best_score = score
                    best_intent = intent
    
    logger.debug(f"Classified intent: {best_intent.value} (confidence: {best_score:.2f})")
    
    # LLM Fallback for classification
    if best_intent == Intent.UNKNOWN or best_score < 0.5:
        try:
            from llm_manager.llm_interface import get_llm_completion
            
            # Simple classification prompt
            intents = [i.value for i in Intent if i != Intent.UNKNOWN]
            prompt = f"""Classify this query into one of these intents: {", ".join(intents)}.
Query: "{query}"
Return ONLY the intent name (or 'unknown')."""

            response = get_llm_completion(prompt, max_tokens=10, temperature=0.0)
            if response:
                cleaned = response.strip().lower().replace('"', '').replace("'", "")
                # Match against enum
                for i in Intent:
                    if i.value == cleaned:
                        logger.info(f"LLM classified intent as: {i.value}")
                        return i, 0.6 # Moderate confidence for LLM
        except Exception as e:
            logger.debug(f"LLM classification failed: {e}")

    return best_intent, best_score


# =============================================================================
# Intent → Plan Templates (Deterministic)
# =============================================================================

def _build_describe_plan(query: str, context: Dict[str, Any]) -> List[PlanStep]:
    """Build plan for data description."""
    return [
        PlanStep(
            step_id="step_1",
            action="profile_data",
            tool="data_profiler",
            inputs={"df": "__context.df__"},
            expected_output="Data profile dictionary",
            fallback_tool="basic_stats",
        ),
    ]


def _build_visualize_plan(query: str, context: Dict[str, Any]) -> List[PlanStep]:
    """Build plan for visualization."""
    # Extract chart type and columns from query
    chart_type = "bar"  # Default
    if re.search(r"\bline\b", query, re.I):
        chart_type = "line"
    elif re.search(r"\bscatter\b", query, re.I):
        chart_type = "scatter"
    elif re.search(r"\bhistogram\b", query, re.I):
        chart_type = "histogram"
    elif re.search(r"\bpie\b", query, re.I):
        chart_type = "pie"
    elif re.search(r"\bheatmap|correlation\b", query, re.I):
        chart_type = "heatmap"
    
    return [
        PlanStep(
            step_id="step_1",
            action="generate_chart",
            tool="chart_generator",
            inputs={
                "df": "__context.df__",
                "chart_type": chart_type,
                "x": "__infer_x__",
                "y": "__infer_y__",
            },
            expected_output="Plotly figure",
            fallback_tool="table_display",
        ),
    ]


def _build_filter_plan(query: str, context: Dict[str, Any]) -> List[PlanStep]:
    """Build plan for filtering."""
    return [
        PlanStep(
            step_id="step_1",
            action="filter_data",
            tool="filter_rows",
            inputs={
                "df": "__context.df__",
                "column": "__infer_column__",
                "operator": "__infer_operator__",
                "value": "__infer_value__",
            },
            expected_output="Filtered DataFrame",
        ),
    ]


def _build_aggregate_plan(query: str, context: Dict[str, Any]) -> List[PlanStep]:
    """Build plan for aggregation."""
    return [
        PlanStep(
            step_id="step_1",
            action="aggregate_data",
            tool="group_by",
            inputs={
                "df": "__context.df__",
                "group_cols": "__infer_group_cols__",
                "agg_dict": "__infer_agg_dict__",
            },
            expected_output="Aggregated DataFrame",
        ),
    ]


def _build_transform_plan(query: str, context: Dict[str, Any]) -> List[PlanStep]:
    """Build plan for transformation."""
    steps = []
    
    # Check for missing value handling
    if re.search(r"\bmissing|null|na|nan\b", query, re.I):
        steps.append(PlanStep(
            step_id="step_1",
            action="fill_missing",
            tool="fill_missing",
            inputs={
                "df": "__context.df__",
                "strategy": "mean",
            },
            expected_output="DataFrame with filled values",
        ))
    else:
        # Generic transform via LLM
        steps.append(PlanStep(
            step_id="step_1",
            action="transform_data",
            tool="pandas_transform",
            inputs={
                "df": "__context.df__",
                "operations": "__llm_generate__",
            },
            expected_output="Transformed DataFrame",
        ))
    
    return steps


PLAN_BUILDERS = {
    Intent.DESCRIBE_DATA: _build_describe_plan,
    Intent.VISUALIZE: _build_visualize_plan,
    Intent.FILTER: _build_filter_plan,
    Intent.AGGREGATE: _build_aggregate_plan,
    Intent.TRANSFORM: _build_transform_plan,
}


# =============================================================================
# Cascade Planner
# =============================================================================

class CascadePlanner:
    """
    Structured cascade planner implementing:
    - Deterministic rules for common intents
    - LLM fallback for novel cases
    - Execution with retry and feedback
    - Learning from executions
    """
    
    MAX_RETRIES = 3
    
    def __init__(self):
        self._execution_history: List[ExecutionResult] = []
    
    def plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Generate an execution plan for the query.
        
        Args:
            query: User's natural language query
            context: Execution context (df, target_col, etc.)
            
        Returns:
            ExecutionPlan ready for execution
        """
        context = context or {}
        
        # Step 1: Classify intent (deterministic first)
        intent, confidence = classify_intent(query)
        
        # Step 2: Build plan from templates
        # Step 1.5: Check for learned plan (Closed Loop)
        steps = []
        try:
            from orchestration.plan_learner import get_learner
            learner = get_learner()
            learned_pattern = learner.get_suggested_plan(intent.value, query)
            if learned_pattern:
                steps = self._create_plan_from_pattern(learned_pattern, context)
                logger.info(f"Using learned plan for intent: {intent.value}")
        except Exception as e:
            logger.warning(f"Failed to retrieve learned plan: {e}")

        if not steps:
            # Step 2: Build plan from templates
            if intent in PLAN_BUILDERS and confidence >= 0.6:
                steps = PLAN_BUILDERS[intent](query, context)
                logger.info(f"Built deterministic plan for intent: {intent.value}")
            else:
                # Fallback: use LLM to generate plan
                steps = self._llm_generate_plan(query, context)
                logger.info(f"Used LLM to generate plan for intent: {intent.value}")
        
        # Generate plan ID
        plan_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            intent=intent,
            query=query,
            steps=steps,
            metadata={
                "confidence": confidence,
                "context_keys": list(context.keys()),
            },
        )
        
        logger.debug(f"Created plan {plan_id} with {len(steps)} steps")
        return plan
    
    def execute(self, plan: ExecutionPlan, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a plan with retry logic and feedback.
        
        Args:
            plan: The execution plan
            context: Runtime context (df, etc.)
            
        Returns:
            ExecutionResult with output or error
        """
        from orchestration.tool_registry import invoke_tool
        
        context = context or {}
        plan.status = PlanStatus.RUNNING
        start_time = datetime.now()
        
        step_outputs = {}  # Store outputs for dependent steps
        resolved_inputs_map = {} 
        last_output = None
        steps_completed = 0
        
        # Index steps for dependency checking
        step_map = {s.step_id: s for s in plan.steps}

        for step in plan.steps:
            # 1. Dependency Check
            if step.depends_on:
                failed_deps = [d for d in step.depends_on if step_map.get(d) and step_map[d].status != PlanStatus.SUCCESS]
                if failed_deps:
                    step.status = PlanStatus.FAILED
                    step.error = f"Dependencies failed: {failed_deps}"
                    logger.warning(f"Skipping step {step.step_id} due to failed dependencies: {failed_deps}")
                    continue

            step.status = PlanStatus.RUNNING
            
            # Resolve inputs
            step_outputs_dict = {k: v for k, v in step_outputs.items()} # Copy safe dict
            resolved_inputs = self._resolve_inputs(step.inputs, context, step_outputs_dict)
            
            # Check for unresolved placeholders
            unresolved = [k for k, v in resolved_inputs.items() if isinstance(v, str) and v.startswith("__")]
            if unresolved:
                # Need LLM to resolve
                resolved_inputs = self._llm_resolve_inputs(step, resolved_inputs, context, plan.intent.value, query=plan.query)
            
            resolved_inputs_map[step.step_id] = resolved_inputs
            
            # Semantic Validation (Auto-fix columns)
            self._validate_semantic_inputs(step.tool, resolved_inputs, context)

            # Execute with retry
            result = invoke_tool(
                step.tool,
                resolved_inputs,
                max_retries=self.MAX_RETRIES,
            )
            
            # 2. Step-Level Fallback
            if not result.success and step.fallback_tool:
                logger.info(f"Step {step.step_id} failed ({result.error}), trying fallback: {step.fallback_tool}")
                # Retry with fallback tool
                fallback_result = invoke_tool(
                    step.fallback_tool,
                    resolved_inputs,
                    max_retries=1, # One try for fallback
                )
                
                if fallback_result.success:
                    logger.info(f"Fallback {step.fallback_tool} succeeded")
                    result = fallback_result
                    step.metadata["fallback_used"] = step.fallback_tool
                    # Note: we don't update step.tool to preserve original intent in log, 
                    # but maybe we should? For now keeping metadata.

            step.execution_time_ms = result.execution_time_ms
            
            if result.success:
                step.status = PlanStatus.SUCCESS
                step.result = result.output
                step_outputs[step.step_id] = result.output
                last_output = result.output
                steps_completed += 1
            else:
                step.status = PlanStatus.FAILED
                step.error = result.error
                logger.warning(f"Step {step.step_id} failed: {result.error}")
                # Do not break; allow dependency check to skip dependent steps
                # while independent steps can proceed.
        
        # Determine overall status
        total_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        if steps_completed == len(plan.steps):
            plan.status = PlanStatus.SUCCESS
            success = True
        elif steps_completed > 0:
            plan.status = PlanStatus.PARTIAL
            success = True  # Partial success
        else:
            plan.status = PlanStatus.FAILED
            success = False
        
        exec_result = ExecutionResult(
            plan_id=plan.plan_id,
            success=success,
            output=last_output,
            error=plan.steps[-1].error if plan.status == PlanStatus.FAILED else None,
            steps_completed=steps_completed,
            total_steps=len(plan.steps),
            execution_time_ms=total_time,
        )
        
        # Log for learning (passive + active)
        self._log_execution(plan, exec_result, context, resolved_inputs_map)
        
        self._execution_history.append(exec_result)
        return exec_result
    
    def _resolve_inputs(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        step_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve input placeholders."""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, str):
                if value == "__context.df__":
                    resolved[key] = context.get("df")
                elif value == "__context.target__":
                    resolved[key] = context.get("target_col")
                elif value.startswith("__step."):
                    # Reference to previous step output
                    ref_step = value.replace("__step.", "").replace("__", "")
                    resolved[key] = step_outputs.get(ref_step)
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    def _llm_generate_plan(self, query: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Use LLM to generate a plan for unknown intents."""
        try:
            from llm_manager.llm_interface import get_llm_completion
            from orchestration.tool_registry import TOOL_REGISTRY
            import json
            import re
            
            # 1. Build context
            df = context.get("df")
            data_context = "No data available."
            if df is not None:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    cols = list(df.columns)
                    dtypes = {k: str(v) for k, v in df.dtypes.items()}
                    data_context = f"Columns: {cols}\nData Types: {dtypes}\nShape: {df.shape}"
            
            # 2. Build tool descriptions
            tools_desc = []
            for name, tool in TOOL_REGISTRY.items():
                if tool.granularity.value == "coarse": # Prefer coarse tools for LLM
                     # simplified schema for prompt
                    schema = {k: v.get("type", "any") for k, v in tool.input_schema.items() if k != "df"}
                    tools_desc.append(f"- {name}: {tool.description}. Inputs: {schema}")
            
            tools_str = "\n".join(tools_desc)
            
            # 3. Construct prompt
            prompt = f"""You are an intelligent data analysis planner. Create a execution plan for the user's query.

AVAILABLE TOOLS:
{tools_str}

DATA CONTEXT:
{data_context}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Select appropriate tools.
2. Use "__context.df__" for dataframe inputs.
3. Return a valid JSON list of steps.

JSON FORMAT:
[
  {{
    "tool": "tool_name",
    "inputs": {{ "arg": "val", "df": "__context.df__" }},
    "reasoning": "step explanation"
  }}
]

JSON RESPONSE:"""

            # 4. Call LLM
            response = get_llm_completion(prompt, max_tokens=1000, temperature=0.1)
            
            if not response:
                logger.warning("Empty LLM response for plan generation")
                return self._fallback_plan()

            # 5. Parse JSON
            try:
                # Extract JSON if wrapped in markdown
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                plan_data = json.loads(json_str.strip())
                
                steps = []
                for i, step_data in enumerate(plan_data):
                    tool_name = step_data.get("tool")
                    if tool_name not in TOOL_REGISTRY:
                        continue
                        
                    steps.append(PlanStep(
                        step_id=f"step_{i+1}",
                        action=tool_name,
                        tool=tool_name,
                        inputs=step_data.get("inputs", {}),
                        expected_output=step_data.get("reasoning", "LLM generated step")
                    ))
                
                if steps:
                    return steps
                    
            except Exception as e:
                logger.warning(f"Failed to parse LLM plan: {e}. Response: {response[:100]}")
                
        except Exception as e:
            logger.error(f"Error in LLM plan generation: {e}")
            
        # Fallback if anything fails
        logger.info("Falling back to default describe plan")
        return self._fallback_plan()

    def _fallback_plan(self) -> List[PlanStep]:
        """Return a safe fallback plan."""
        return [
            PlanStep(
                step_id="step_1",
                action="describe_data",
                tool="data_profiler",
                inputs={"df": "__context.df__"},
                expected_output="Data profile",
            ),
        ]
    
    def _llm_resolve_inputs(
        self,
        step: PlanStep,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        intent: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use learned patterns (and eventually LLM) to resolve placeholder inputs.
        """
        resolved = dict(inputs)
        df = context.get("df")
        
        if df is not None:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                columns = list(df.columns)
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                
                # 1. Try learned inputs first
                if intent and query:
                    try:
                        from orchestration.plan_learner import get_learner
                        learner = get_learner()
                        # Pass query to find specific pattern match
                        suggested = learner.get_suggested_inputs(intent, context, query=query)
                        
                        for key, val in suggested.items():
                            # Only apply if we have a placeholder for this key
                            current_val = resolved.get(key)
                            if isinstance(current_val, str) and current_val.startswith("__"):
                                # Verify the suggested column actually exists in this DF
                                if val in columns:
                                    resolved[key] = val
                                    logger.debug(f"Resolved '{key}' to '{val}' using learned pattern")
                    except Exception as e:
                        logger.warning(f"Failed to apply learned inputs: {e}")

                # 2. Fallback to heuristics
                if "__infer_x__" in str(resolved.get("x", "")):
                    resolved["x"] = columns[0] if columns else None
                    
                if "__infer_y__" in str(resolved.get("y", "")):
                    resolved["y"] = numeric_cols[0] if numeric_cols else None
                
                if "__infer_column__" in str(resolved.get("column", "")):
                    resolved["column"] = columns[0] if columns else None
                
                if "__infer_group_cols__" in str(resolved.get("group_cols", "")):
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    resolved["group_cols"] = cat_cols[:1] if cat_cols else columns[:1]
                
                if "__infer_agg_dict__" in str(resolved.get("agg_dict", "")):
                    if numeric_cols:
                        resolved["agg_dict"] = {numeric_cols[0]: "sum"}
                
                # 3. Filter Resolution
                if "__infer_operator__" in str(resolved.get("operator", "")) or "__infer_value__" in str(resolved.get("value", "")):
                    try:
                        filter_params = self._infer_filter_from_query(query, df)
                        
                        # Trust the inferred column if we found one
                        # This fixes cases where LLM guesses the wrong column (e.g. Date)
                        if filter_params.get("column"):
                            resolved["column"] = filter_params.get("column")
                            
                        resolved["operator"] = filter_params.get("operator")
                        resolved["value"] = filter_params.get("value")
                    except Exception as e:
                        logger.warning(f"Filter inference failed: {e}")

                # 4. Transform Resolution
                if "__llm_generate__" in str(resolved.get("operations", "")):
                    resolved["operations"] = self._generate_transform_ops(query, df)

        return resolved

    def _infer_filter_from_query(self, query: str, df: Any) -> Dict[str, Any]:
        """Infer filter column, operator, and value from query."""
        import re
        q = query.lower()
        columns = list(df.columns) if hasattr(df, "columns") else []
        
        # 1. Identify Column
        # Sort columns by length descent to match longest first
        sorted_cols = sorted(columns, key=len, reverse=True)
        target_col = None
        for col in sorted_cols:
            if col.lower() in q:
                target_col = col
                break
        
        # Do NOT default to first column blindly. 
        # If no column found, return None for column so we fallback to LLM's guess (or fail gracefully)
        
        # 2. Identify Operator & Value
        operator = "=="
        value = 0
        
        # Map phrases to operators
        op_map = {
            r"greater than|above|over|exceeds?": ">",
            r"less than|below|under": "<",
            r"at least|min(imum)?": ">=",
            r"at most|max(imum)?": "<=",
            r"equals?|is": "==",
            r"not equals?|is not|different": "!=",
            r"contains?|includes?|has": "contains"
        }
        
        # Regex for values (int/float)
        val_pattern = r"[-+]?\d*\.?\d+"
        
        # Search for "operator value" pattern
        found_op = False
        for pattern, op_sym in op_map.items():
            # Look for "operator ... value"
            full_pattern = f"({pattern}).*?({val_pattern})"
            match = re.search(full_pattern, q)
            if match:
                operator = op_sym
                val_str = match.group(2)
                value = float(val_str) if "." in val_str else int(val_str)
                found_op = True
                break
        
        # Fallback: check for symbolic operators
        if not found_op:
            sym_match = re.search(f"(>=|<=|!=|==|>|<).*?({val_pattern})", q)
            if sym_match:
                operator = sym_match.group(1)
                val_str = sym_match.group(2)
                value = float(val_str) if "." in val_str else int(val_str)
        
        return {"column": target_col, "operator": operator, "value": value}

    def _generate_transform_ops(self, query: str, df: Any) -> List[Dict[str, Any]]:
        """Generate transform operations list using LLM."""
        try:
            from llm_manager.llm_interface import get_llm_completion
            import json
            
            cols_info = list(df.columns) if hasattr(df, "columns") else "unknown"
            
            prompt = f"""Generate a list of pandas data transformation operations for this query.
Query: "{query}"
Columns: {cols_info}

Supported operations:
- fill_missing (params: strategy="mean"|"median"|"mode"|"custom", value=?)
- drop_columns (params: columns=[list])
- rename_columns (params: mapping={{old: new}})
- filter_rows (params: condition="query string")
- sort_values (params: by=col, ascending=True/False)

Return ONLY a valid JSON list of operations.
Example: [{{"operation": "fill_missing", "params": {{"strategy": "mean"}}}}]
"""
            response = get_llm_completion(prompt, max_tokens=500)
            if not response: 
                return []
                
            # Extract JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            ops = json.loads(json_str.strip())
            return ops if isinstance(ops, list) else []
            
        except Exception as e:
            logger.error(f"Failed to generate transform ops: {e}")
            return []
    
    def _validate_semantic_inputs(self, tool_name: str, inputs: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Validate inputs against context and auto-correct typos."""
        error = ""
        df = inputs.get("df")
        
        # Check for DataFrame columns
        if hasattr(df, "columns"):
            columns = set(df.columns)
            potential_cols = []
            
            # Identify arguments that should be columns
            if "x" in inputs: potential_cols.append(("x", inputs["x"]))
            if "y" in inputs: potential_cols.append(("y", inputs["y"]))
            if "column" in inputs: potential_cols.append(("column", inputs["column"]))
            if "by" in inputs: # sort/group
                 val = inputs["by"]
                 if isinstance(val, list):
                     for c in val: potential_cols.append(("by", c))
                 elif isinstance(val, str):
                     potential_cols.append(("by", val))

            for arg_name, col_name in potential_cols:
                # If col_name matches a real column, good.
                if col_name in columns:
                    continue
                    
                # If string and not found -> try fuzzy match
                if isinstance(col_name, str):
                    import difflib
                    matches = difflib.get_close_matches(col_name, list(columns), n=1, cutoff=0.8)
                    if matches:
                        logger.info(f"Auto-correcting column '{col_name}' to '{matches[0]}'")
                        # Fix in inputs
                        if arg_name == "by" and isinstance(inputs["by"], list):
                             # tricky to update list in place without index
                             idx = inputs["by"].index(col_name)
                             inputs["by"][idx] = matches[0]
                        else:
                             inputs[arg_name] = matches[0]
                    else:
                        logger.warning(f"Column '{col_name}' not found for tool '{tool_name}' argument '{arg_name}'")
        return error

    def _create_plan_from_pattern(self, pattern: Any, context: Dict[str, Any]) -> List[PlanStep]:
        """Create steps from a learned pattern."""
        steps = []
        # Eagerly apply all learned inputs to all steps (tools ignore extra kwargs)
        learned_inputs = pattern.input_mappings.copy()
        
        for i, tool_name in enumerate(pattern.tool_sequence):
            step_id = f"step_{i+1}"
            step_inputs = learned_inputs.copy()
            step_inputs["df"] = "__context.df__" # Always provide df reference
            
            steps.append(PlanStep(
                step_id=step_id,
                action=tool_name,
                tool=tool_name,
                inputs=step_inputs,
                expected_output=f"Auto-generated step for {tool_name}"
            ))
            
        return steps

    def _log_execution(
        self, 
        plan: ExecutionPlan, 
        result: ExecutionResult, 
        context: Optional[Dict[str, Any]] = None,
        resolved_inputs_map: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Log execution for learning system (passive + active)."""
        context = context or {}
        resolved_inputs_map = resolved_inputs_map or {}
        
        # 1. Passive logging to interaction logger
        try:
            from llm_learning.interaction_logger import get_interaction_logger, InteractionType
            
            il = get_interaction_logger()
            il.log(
                prompt=plan.query,
                response=json.dumps(result.output)[:1000] if result.output else "",
                interaction_type=InteractionType.ACTION,
                code_generated="",
                execution_success=result.success,
                response_time_ms=result.execution_time_ms,
            )
            logger.debug(f"Logged execution to InteractionLogger for plan {plan.plan_id}")
        except Exception as e:
            logger.debug(f"Could not log to InteractionLogger: {e}")
        
        # 2. Active learning: update patterns and tool weights
        try:
            from orchestration.plan_learner import get_learner
            
            learner = get_learner()
            learner.learn_from_execution(plan, result, context, resolved_inputs_map)
            result.learned = True
            logger.debug(f"Active learning applied for plan {plan.plan_id}")
        except Exception as e:
            logger.debug(f"Could not apply active learning: {e}")
    
    def get_history(self, limit: int = 10) -> List[ExecutionResult]:
        """Get recent execution history."""
        return self._execution_history[-limit:]


# =============================================================================
# Global Access
# =============================================================================

_planner: Optional[CascadePlanner] = None


def get_planner() -> CascadePlanner:
    """Get the global cascade planner instance."""
    global _planner
    if _planner is None:
        _planner = CascadePlanner()
    return _planner
