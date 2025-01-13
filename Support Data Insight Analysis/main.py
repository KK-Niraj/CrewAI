# Warning control
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
from helper import load_env

load_env()

import crewai
import os
import yaml
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool
from unittest.mock import patch


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
csv_tool = FileReadTool(file_path="./support_tickets_data.csv")

# Creating Agents
suggestion_generation_agent = Agent(
    config=agents_config["suggestion_generation_agent"], tools=[csv_tool]
)

reporting_agent = Agent(config=agents_config["reporting_agent"], tools=[csv_tool])


with patch("crewai.agent.Agent._validate_docker_installation"):
    chart_generation_agent = Agent(
        config=agents_config["chart_generation_agent"], allow_code_execution=True
    )

# Creating Tasks
suggestion_generation = Task(
    config=tasks_config["suggestion_generation"], agent=suggestion_generation_agent
)

table_generation = Task(config=tasks_config["table_generation"], agent=reporting_agent)

chart_generation = Task(
    config=tasks_config["chart_generation"], agent=chart_generation_agent
)

final_report_assembly = Task(
    config=tasks_config["final_report_assembly"],
    agent=reporting_agent,
    context=[suggestion_generation, table_generation, chart_generation],
)


# Creating Crew
support_report_crew = Crew(
    agents=[suggestion_generation_agent, reporting_agent, chart_generation_agent],
    tasks=[
        suggestion_generation,
        table_generation,
        chart_generation,
        final_report_assembly,
    ],
    verbose=True,
)
with patch.object(
    crewai.agent.Agent, "_validate_docker_installation", return_value=None
):
    support_report_crew.test(n_iterations=1, openai_model_name="gpt-4o")
