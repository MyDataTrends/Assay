"""
Scheduler Agent - Manages recurring analysis tasks.

Capabilities:
- Schedule analysis workflows (e.g., "Run Sales Report every Monday")
- cron-like execution of agent tasks
- Persist schedules to local JSON/DB
- Trigger Email/PDF delivery
"""
import time
import threading
import schedule
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SchedulerAgent:
    def __init__(self, storage_path: str = "local_data/schedules.json"):
        self.storage_path = Path(storage_path)
        self.running = False
        self.jobs = []
        self._load_schedules()

    def _load_schedules(self):
        """Load existing schedules from disk."""
        if not self.storage_path.exists():
            self._save_schedules([])
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for job_data in data:
                    self.add_job(job_data, save=False)
        except Exception as e:
            logger.error(f"Failed to load schedules: {e}")

    def _save_schedules(self, jobs_data: List[Dict]):
        """Persist schedules to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(jobs_data, f, indent=2)

    def add_job(self, job_config: Dict[str, Any], save: bool = True):
        """
        Add a new job to the scheduler.
        
        job_config:
        {
            "id": str,
            "name": str,
            "frequency": "daily" | "weekly" | "interval",
            "interval_minutes": int (optional),
            "day_of_week": "monday" (optional),
            "time": "HH:MM" (24h),
            "dataset_path": str,
            "prompt": str
        }
        """
        self.jobs.append(job_config)
        
        # Schedule the job
        self._schedule_job(job_config)
        
        logger.info(f"Scheduled job added: {job_config['name']}")
        
        if save:
            self._save_schedules(self.jobs)

    def _schedule_job(self, job: Dict[str, Any]):
        """Register the job with the schedule library."""
        try:
            freq = job.get("frequency", "daily")
            time_str = job.get("time", "09:00")
            
            runner = schedule.every()
            
            if freq == "interval":
                minutes = job.get("interval_minutes", 60)
                runner = runner.minutes.at(f":{minutes:02d}") # Syntax differs slightly for interval, simplified:
                runner = schedule.every(minutes).minutes
            elif freq == "weekly":
                day = job.get("day_of_week", "monday").lower()
                runner = getattr(runner, day).at(time_str)
            else: # daily
                runner = runner.day.at(time_str)
                
            runner.do(self._execute_job, job)
            
        except Exception as e:
            logger.error(f"Failed to schedule job {job['name']}: {e}")

    def _execute_job(self, job: Dict[str, Any]):
        """Execute the analysis job."""
        logger.info(f"Executing job: {job['name']}")
        try:
            import pandas as pd
            from orchestration.cascade_planner import CascadePlanner
            from ui.chat_logic import generate_natural_answer
            
            # 1. Load Data
            path = Path(job["dataset_path"])
            if not path.exists():
                logger.error(f"Dataset not found: {path}")
                return
                
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix in [".parquet"]:
                df = pd.read_parquet(path)
            else:
                logger.error("Unsupported file format")
                return

            # 2. Plan & Execute (Robust Cascade)
            context = {"df": df, "job_name": job['name']}
            planner = CascadePlanner()
            
            logger.info(f"Generating plan for: {job['prompt']}")
            plan = planner.plan(job["prompt"], context=context)
            
            logger.info(f"Executing plan {plan.plan_id} ({len(plan.steps)} steps)")
            exec_result = planner.execute(plan, context=context)
            
            if not exec_result.success:
                logger.error(f"Execution failed: {exec_result.error}")
                result_str = f"Error executing analysis: {exec_result.error}"
                result = None
            else:
                raw_result = exec_result.output
                
                # Sanitize numpy types to prevent them from leaking into the report
                def _sanitize(obj):
                    if isinstance(obj, dict):
                        return {k: _sanitize(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [_sanitize(v) for v in obj]
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    return obj
                    
                result = _sanitize(raw_result)
                result_str = str(result)
                
            # 3. Narrative
            narrative = generate_natural_answer(job["prompt"], result)
            
            # 5. Save Report
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"{job['id']}_{timestamp}.md"
            
            report_content = f"""# {job['name']}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Dataset:** {path.name}

## Objective
{job['prompt']}

## Analysis Result
{narrative}

## Raw Data
```
{result_str}
```
"""
            with open(report_file, 'w') as f:
                f.write(report_content)
                
            logger.info(f"Report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Job execution failed: {e}")

    def list_jobs(self) -> List[Dict]:
        return self.jobs

    def start(self):
        """Start the scheduler loop in a background thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Scheduler started.")

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        logger.info("Scheduler stopped.")

    def _run_loop(self):
        while self.running:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    agent = SchedulerAgent()
    
    # Add a test job
    job = {
        "id": "test_1",
        "name": "Test Job",
        "frequency": "interval",
        "interval_minutes": 1,
        "dataset_path": "User_Data/sample_user/user_data.csv", # Update to valid path
        "prompt": "What is the average sales?"
    }
    agent.add_job(job, save=False)
    
    agent.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
