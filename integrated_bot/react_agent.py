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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Optional, List, Dict, Union

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define indoor locations and their coordinates
indoor_locations = {
    "kitchen": [3, -1, 45],
    "living room": [0, 0, 90],
    "bedroom": [6.7, 1.4, 180],
    "bathroom": [3, 4, 90],
    "study": [4, -2, 0]
}

# Define output parsers using Pydantic models
class Coordinates(BaseModel):
    x: float
    y: float
    theta: float
    location: str
    message: str

class LocationsList(BaseModel):
    locations: List[str]
    message: str

class Goal(BaseModel):
    x: float
    y: float
    theta: float

class GoalResponse(BaseModel):
    success: bool
    message: str
    goal: Goal

# Define tools with structured outputs
@tool
def get_coordinates(indoor_location: str) -> Coordinates:
    """Get coordinates for a given indoor location."""
    message = f"Tool 'get_coordinates' called with argument: {indoor_location}"
    logger.info(message)
    location = indoor_location.lower()
    
    if location in indoor_locations:
        x, y, theta = indoor_locations[location]
        return Coordinates(
            x=x,
            y=y,
            theta=theta,
            location=location,
            message=f"Found coordinates for {location}"
        )
    else:
        raise ValueError(f"Location '{location}' not found")

@tool
def list_locations() -> LocationsList:
    """Get a list of all available indoor locations."""
    message = "Tool 'list_locations' called"
    logger.info(message)
    return LocationsList(
        locations=list(indoor_locations.keys()),
        message="Retrieved available locations"
    )

@tool
async def set_goal(goal: Goal) -> GoalResponse:
    """Set a navigation goal for the robot."""
    message = f"Tool 'set_goal' called with argument: x={goal.x}, y={goal.y}, theta={goal.theta}"
    logger.info(message)
    
    try:
        subprocess.run(["ros2", "run", "turtlebot3_web_nav", "ros2_bridge", "set_goal", 
                       str(goal.x), str(goal.y), str(goal.theta)])
        return GoalResponse(
            success=True,
            message=f"Goal set successfully",
            goal=goal
        )
    except Exception as e:
        return GoalResponse(
            success=False,
            message=f"Failed to set goal: {str(e)}",
            goal=goal
        )

tools = [get_coordinates, list_locations, set_goal]

# Define conversation history tracking
class ConversationState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = Field(default_factory=list)
    tool_outputs: List[Dict] = Field(default_factory=list)

# Define system prompt
SYSTEM_PROMPT_1 = """You are an AI assistant specifically designed for indoor navigation of a TurtleBot3 robot. 
Your primary function is to help users navigate the robot to valid indoor locations within the building.

IMPORTANT VALIDATION RULES:
1. For ANY navigation request, your FIRST step must ALWAYS be to check available locations using the list_locations tool.
2. After checking list_locations, you must:
   a. If the requested location IS in the list:
      - Proceed with getting coordinates and setting the goal
   b. If the requested location is NOT in the list:
      - Stop immediately
      - Do not proceed with any further steps
      - Explain to the user that only indoor locations from the list are valid
      - Show them the available locations they can choose from

PLANNING APPROACH:
For "go to kitchen" (valid location):
1. Check available locations using list_locations
2. Get coordinates for kitchen using get_coordinates
3. Set navigation goal using set_goal

For "go to Antarctica" (invalid location):
1. Check available locations using list_locations
2. Stop and inform user that Antarctica is not a valid indoor location
3. List the available locations they can navigate to

Remember:
- Never plan steps for getting coordinates or setting goals for invalid locations
- Always verify location validity before proceeding with navigation
- Keep responses focused on indoor navigation within the building
- Provide helpful guidance when users request invalid locations

After list_locations:
✓ For valid locations: Continue with next steps
✗ For invalid locations: Stop and provide guidance"""

SYSTEM_PROMPT_2 = """For the given objective, come up with a simple step by step plan. \
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
                    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""

# Create the agent with proper prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_1),
    ("system", SYSTEM_PROMPT_2),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools)

# Define the State with conversation tracking
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    conversation: ConversationState

# Planning Step
class Plan(BaseModel):
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_1),
    ("system", SYSTEM_PROMPT_2),
    ("placeholder", "{messages}"),
])
planner = planner_prompt | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Plan)

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    if "conversation" not in state:
        state["conversation"] = ConversationState()
    state["conversation"].messages.append(
        HumanMessage(content=state["input"])
    )
    return {"plan": plan.steps}

# Execute Step with Conversation Tracking
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    task = plan[0]
    
    tool_selection_message = AIMessage(content=f"I will execute: {task}")
    state["conversation"].messages.append(tool_selection_message)
    
    agent_response = await agent_executor.ainvoke({
        "messages": [("user", task)]
    })
    
    tool_output = agent_response["messages"][-1].content
    state["conversation"].tool_outputs.append({
        "task": task,
        "output": tool_output
    })
    
    tool_response_message = AIMessage(
        content=tool_output,
        additional_kwargs={"role": "tool"}
    )
    state["conversation"].messages.append(tool_response_message)
    
    return {
        "past_steps": [(task, tool_output)],
        "conversation": state["conversation"]
    }

# Replan Step
class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add superfluous steps. \
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

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

# Define end condition
def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

# Create the Graph
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

# Main controlling logic
async def run_react_agent(user_input: str) -> str:
    inputs = {"input": user_input, "conversation": ConversationState()}
    formatted_responses = []
    planning_steps = []
    
    try:
        async for event in app.astream(inputs, config={"recursion_limit": 50}):
            logger.info(f"Processing event: {event}")
            
            # Handle planner events
            if "planner" in event:
                plan = event["planner"].get("plan", [])
                if plan:
                    planning_steps = [f"🔹 {step}" for step in plan]
                    formatted_responses.append("Planning Steps:")
                    formatted_responses.extend(planning_steps)
                    formatted_responses.append("")  # Add blank line
            
            # Handle conversation state
            if "conversation" in event:
                conv_state = event["conversation"]
                current_tool_execution = []
                
                for message in conv_state.messages:
                    if isinstance(message, AIMessage):
                        role = message.additional_kwargs.get("role", "assistant")
                        content = message.content.strip()
                        
                        if content:
                            if role == "tool":
                                current_tool_execution.append(f"🛠️ Result: {content}")
                            elif content.startswith("I will execute:"):
                                current_tool_execution.append(f"⚙️ Action: {content}")
                            else:
                                current_tool_execution.append(f"🤖 {content}")
                
                if current_tool_execution:
                    formatted_responses.extend(current_tool_execution)
                    formatted_responses.append("")  # Add blank line
            
            # Handle tool outputs directly from event
            if "agent" in event:
                agent_data = event["agent"]
                if "past_steps" in agent_data:
                    for step, result in agent_data["past_steps"]:
                        formatted_responses.append(f"📋 Step: {step}")
                        formatted_responses.append(f"✅ Result: {result}")
                        formatted_responses.append("")  # Add blank line
            
            # Handle replan response
            if "replan" in event:
                replan_data = event["replan"]
                if "response" in replan_data:
                    formatted_responses.append(f"🎯 Final Status: {replan_data['response']}")
                    formatted_responses.append("")  # Add blank line
                elif "plan" in replan_data:
                    new_steps = [f"📝 Next Step: {step}" for step in replan_data["plan"]]
                    formatted_responses.extend(new_steps)
                    formatted_responses.append("")  # Add blank line

        # Format final response with proper structure
        response_sections = []
        current_section = []
        
        for line in formatted_responses:
            if not line.strip():  # Empty line indicates section break
                if current_section:
                    response_sections.append("\n".join(current_section))
                    current_section = []
            else:
                current_section.append(line)
        
        # Add any remaining section
        if current_section:
            response_sections.append("\n".join(current_section))
        
        # Join all sections with clear separation
        final_response = "\n\n".join(filter(None, response_sections))
        
        # Ensure non-empty response
        if not final_response.strip():
            final_response = "I understood your request but couldn't generate a proper response. Please try again."
            logger.warning("No responses generated for input: %s", user_input)
        
        return final_response
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        return error_msg