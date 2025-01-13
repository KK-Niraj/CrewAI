# Warning control
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
from helper import load_env

load_env()

import os
import pandas as pd
import yaml
from crewai import Agent, Task, Crew
from schema import ProjectPlan

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"


def automated_projects():
    # Define file paths for YAML configurations
    files = {
        'lead_agents': 'config/lead_qualification_agents.yaml',
        'lead_tasks': 'config/lead_qualification_tasks.yaml',
        'email_agents': 'config/email_engagement_agents.yaml',
        'email_tasks': 'config/email_engagement_tasks.yaml'
    }

    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    lead_agents_config = configs['lead_agents']
    lead_tasks_config = configs['lead_tasks']
    email_agents_config = configs['email_agents']
    email_tasks_config = configs['email_tasks']

    # Creating Agents
    project_planning_agent = Agent(config=agents_config["project_planning_agent"])

    estimation_agent = Agent(config=agents_config["estimation_agent"])

    resource_allocation_agent = Agent(config=agents_config["resource_allocation_agent"])

    # Creating Tasks
    task_breakdown = Task(
        config=tasks_config["task_breakdown"], agent=project_planning_agent
    )

    time_resource_estimation = Task(
        config=tasks_config["time_resource_estimation"], agent=estimation_agent
    )

    resource_allocation = Task(
        config=tasks_config["resource_allocation"],
        agent=resource_allocation_agent,
        output_pydantic=ProjectPlan,  # This is the structured output we want
    )

    # Creating Crew
    crew = Crew(
        agents=[project_planning_agent, estimation_agent, resource_allocation_agent],
        tasks=[task_breakdown, time_resource_estimation, resource_allocation],
        verbose=True,
    )

    return crew


if __name__ == "__main__":
    project = "Website"
    industry = "Technology"
    project_objectives = "Create a website for a small business"
    team_members = """
    - John Doe (Project Manager)
    - Jane Doe (Software Engineer)
    - Bob Smith (Designer)
    - Alice Johnson (QA Engineer)
    - Tom Brown (QA Engineer)
    """
    project_requirements = """
    - Create a responsive design that works well on desktop and mobile devices
    - Implement a modern, visually appealing user interface with a clean look
    - Develop a user-friendly navigation system with intuitive menu structure
    - Include an "About Us" page highlighting the company's history and values
    - Design a "Services" page showcasing the business's offerings with descriptions
    - Create a "Contact Us" page with a form and integrated map for communication
    - Implement a blog section for sharing industry news and company updates
    - Ensure fast loading times and optimize for search engines (SEO)
    - Integrate social media links and sharing capabilities
    - Include a testimonials section to showcase customer feedback and build trust
    """
    # The given Python dictionary
    inputs = {
        "project_type": project,
        "project_objectives": project_objectives,
        "industry": industry,
        "team_members": team_members,
        "project_requirements": project_requirements,
    }
    crew = automated_projects()

    # Run the crew
    result = crew.kickoff(inputs=inputs)
    costs = (
        0.150
        * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens)
        / 1_000_000
    )
    print(f"Total costs: ${costs:.4f}")

    # Convert UsageMetrics instance to a DataFrame
    df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
    print(df_usage_metrics)
    print(result.pydantic.dict())
    tasks = result.pydantic.dict()["tasks"]
    df_tasks = pd.DataFrame(tasks)

    # Display the DataFrame as an HTML table
    df_tasks.style.set_table_attributes('border="1"').set_caption(
        "Task Details"
    ).set_table_styles([{"selector": "th, td", "props": [("font-size", "120%")]}])
    print(df_tasks)
    milestones = result.pydantic.dict()["milestones"]
    df_milestones = pd.DataFrame(milestones)

    # Display the DataFrame as an HTML table
    df_milestones.style.set_table_attributes('border="1"').set_caption(
        "Task Details"
    ).set_table_styles([{"selector": "th, td", "props": [("font-size", "120%")]}])
    print(df_milestones)
