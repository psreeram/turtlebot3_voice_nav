# TurtleBot3 Voice Navigation System

The TurtleBot3 Voice Navigation System is designed to enable users to control a TurtleBot3 robot using voice commands or text input through a Telegram bot interface. The system integrates various components, including a Telegram bot, voice transcription, natural language processing, an AI Agent and ROS2 navigation, to provide an intuitive and user-friendly way to navigate the robot in indoor environments.

## System Components

The system consists of the following main components:
- Telegram Bot Interface
- Voice Transcription Service
- Plan-and-Execute AI Agent
- ROS2 Bridge
- TurtleBot3 with ROS2 and Nav2

## System Architecture

![System Architecture](https://cdn-images-1.medium.com/max/800/1*QSsOVL9YzhWKJhNCI1fZsQ.png)

## Detailed Component Descriptions

### 3.1 Telegram Bot Interface (integrated_bot.py)
- Handles user interactions through the Telegram messaging platform
- Processes both voice messages and text input
- Sends voice messages for transcription and text messages directly to the React Agent
- Manages user sessions and provides a welcome message

### 3.2 Voice Transcription Service (voice_transcribe.py)
- Uses the OpenAI Whisper API to transcribe voice messages to text
- Handles the upload of voice data and retrieval of transcribed text

### 3.3 Plan-and-Execute AI Agent (react_agent.py)
- Implements a Plan and Execute agent using LangChain and LLM
- Interprets user commands and generates a plan of action
- Uses custom tools to interact with the ROS2 environment:
  - list_locations: Provides a list of available indoor locations
  - get_coordinates: Retrieves coordinates for predefined indoor locations
  - set_goal: Sets a navigation goal for the robot

### 3.4 ROS2 Bridge (ros2_bridge.py)
- Acts as an interface between the React Agent and the ROS2 environment
- Converts high-level navigation commands into ROS2 messages
- Interacts with the Nav2 stack to send navigation goals
- Provides feedback on the status of navigation tasks

### 3.5 TurtleBot3 with ROS2 and Nav2
- The TurtleBot3 robot (simulation) running ROS2 and the Nav2 navigation stack
- Executes the navigation commands received through the ROS2 Bridge
