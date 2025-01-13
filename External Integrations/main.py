# Warning control
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
from helper import get_openai_api_key

get_openai_api_key()

import os
import json
import yaml
from crewai import Agent, Task, Crew

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"


def automated_projects():
    # Define file paths for YAML configurations
    files = {"agents": "config/agents.yaml", "tasks": "config/tasks.yaml"}

    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, "r") as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs["agents"]
    tasks_config = configs["tasks"]

    # Creating Agents
    data_collection_agent = Agent(
        config=agents_config["data_collection_agent"],
        tools=[BoardDataFetcherTool(), CardDataFetcherTool()],
    )

    analysis_agent = Agent(config=agents_config["analysis_agent"])

    # Creating Tasks
    data_collection = Task(
        config=tasks_config["data_collection"], agent=data_collection_agent
    )

    data_analysis = Task(config=tasks_config["data_analysis"], agent=analysis_agent)

    report_generation = Task(
        config=tasks_config["report_generation"],
        agent=analysis_agent,
    )

    # Creating Crew
    crew = Crew(
        agents=[data_collection_agent, analysis_agent],
        tasks=[data_collection, data_analysis, report_generation],
        verbose=True,
    )

    return crew

if __name__ == "__main__":
    crew = automated_projects()
    # Kick off the crew and execute the process
    result = crew.kickoff()
    costs = 0.150 * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens) / 1_000_000
    print(f"Total costs: ${costs:.4f}")

    # Convert UsageMetrics instance to a DataFrame
    df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
    print(df_usage_metrics)
    markdown  = result.raw
    Markdown(markdown)