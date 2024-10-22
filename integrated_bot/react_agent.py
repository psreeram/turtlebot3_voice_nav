import getpass
import os
import json
import subprocess
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import logging
import operator
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define indoor locations and their coordinates
indoor_locations = {
    "kitchen": [3, -1, 45],
    "living room": [0, 0, 90],
    "bedroom": [6.7, 1.4, 180],
    "bathroom": [3, 4, 90],
    "study": [4, -2, 0]
}

# Define tools
@tool
def get_coordinates(indoor_location: str):
    """Get coordinates for a given indoor location."""
    message = f"Tool 'get_coordinates' called with argument: {indoor_location}"
    print(message)
    location = indoor_location.lower()
    if location in indoor_locations:
        x, y, theta = indoor_locations[location]
        return f"{message}\nCoordinates for {location}: " + json.dumps({"x": x, "y": y, "theta": theta})
    else:
        return f"{message}\nError: Location '{location}' not found"

@tool
def list_locations():
    """Get a list of all available indoor locations."""
    message = "Tool 'list_locations' called"
    print(message)
    return f"{message}\nAvailable locations: " + json.dumps(list(indoor_locations.keys()))

class Goal(BaseModel):
    x: float
    y: float
    theta: float

@tool
async def set_goal(goal: Goal) -> dict:
    """Set a navigation goal for the robot."""
    message = f"Tool 'set_goal' called with argument: x={goal.x}, y={goal.y}, theta={goal.theta}"
    print(message)
    subprocess.run(["ros2", "run", "turtlebot3_web_nav", "ros2_bridge", "set_goal", str(goal.x), str(goal.y), str(goal.theta)])
    return {"message": f"{message}\nGoal set successfully to x={goal.x}, y={goal.y}, theta={goal.theta}"}

tools = [get_coordinates, list_locations, set_goal]
tool_node = ToolNode(tools)

# Define Execution Agent
prompt = hub.pull("ih/ih-react-agent-executor")
prompt = """You are an AI assistant that helps with navigation requests. \
            Your primary function is to use the provided tools to look up coordinates and set navigation goals. \
            It is crucial that you actually call these tools, not just mention them. \
            Follow these steps for every request: \
            1. Interpret if the input is a navigation request. Look for terms like \"go to\", \"navigate to\", \"move to\", etc. \
            2. For navigation requests: \
              a. Extract the location (e.g., \"Kitchen\", \"Living room\"). \
              b. Try to find the coordinates of a location using the given tools."""

llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

# Define the State
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Planning Step
class Plan(BaseModel):
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """For the given objective, come up with a simple step by step plan. \
                This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
                The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""),
    ("system", """You are an AI assistant that helps with navigation requests for CoffeeBot. \
            Your primary function is to use the provided tools to look up coordinates and set navigation goals. \
            It is crucial that you actually call these tools, not just mention them. \
            Follow these steps for every request: \
            1. Interpret if the input is a navigation request. Look for terms like \"go to\", \"navigate to\", \"move to\", etc. \
            2. For navigation requests: \
              a. Extract the location from the command text or input text or transcibed text(e.g., \"Kitchen\", \"Living room\"). \
              b. Try to find the coordinates of a location using the given tools."""),
    ("placeholder", "{messages}"),
])
planner = planner_prompt | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Plan)

# Re-Plan Step
class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Act)

# Create the Graph
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

app = workflow.compile()



# The main controlling logic resides here
async def run_react_agent(user_input: str) -> str:
    inputs = {"input": user_input}
    all_messages: List[str] = []  # List to store all messages in order
    errors: List[str] = []

    try:
        async for event in app.astream(inputs, config={"recursion_limit": 50}):
            logger.info(f"Processing event: {event}")
            for k, v in event.items():
                if k == "__end__":
                    continue
                elif k == "error":
                    error_message = f"Error: {v}"
                    errors.append(error_message)
                    logger.error(error_message)
                elif k.startswith("tool_"):
                    tool_name = k[5:]
                    tool_message = f"Tool '{tool_name}' called: {str(v)}"
                    all_messages.append(tool_message)
                    logger.info(f"Added tool message: {tool_message}")
                else:
                    if isinstance(v, dict):
                        message = json.dumps(v)
                    else:
                        message = str(v)
                    all_messages.append(message)
                    logger.info(f"Added message: {message}")

            logger.info(f"Step completed. Current all_messages: {all_messages}")

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        errors.append(error_message)

    # Format the final response
    final_response = ["Response:"]
    final_response.extend(all_messages)
    final_response.append("")  # Add a blank line after all messages

    if errors:
        final_response.append("Errors encountered:")
        final_response.extend(errors)
        final_response.append("")  # Add a blank line after errors

    formatted_response = "\n".join(final_response)
    logger.info(f"Final response:\n{formatted_response}")
    return formatted_response