The HTN Planner automatically generates detailed plans utilizing OpenAI's GPT
and the Hierarchical Task Network (HTN) architecture.
The system generates tasks to achieve a goal using the LLM and then iteratively
decomposes it into subtasks that can be executed.

For best results use GPT-4, though other OpenSource LLMs may suffice with modifications to the api

Components:
- Decomposition - Takes a task and decomposes it into subtasks until the max depth is reached or the plan has failed.
 The system keeps track of candidate decompositions and attempts to choose the best option. May exit early if results are good.
- Re-planning - When planning fails or part of a plan fails, re-planning occurs
- Task Execution - Identifies a task as an executable unit
  - At present tasks are not actually executed in a terminal
- State Tracking - The LLM tracks and updates the state as execution occurs
- Text Parsing - Parses and extracts information from the natural language responses produced by the LLM
- Task Translation - Attempts to translate a low level task into a command or piece of code that can be executed.
- Frontend - A simple react frontend to display a hierarchy representing the plan
- Logs - A large variety of logs are generated in the "logs" folder and function traces can be found in "function_trace.log"
  - function_trace.log - Tracks all the function calls annotated with "@trace_function_calls"
  - The logs in the "logs" folder each track a particular sub-system using the "log_response" function
  - parsing_errors - Tracks any issues with parsing the output from the LLM so that updates can be made to the parser to fix the issue
  - state_changes - Tracks the state transitions over time generated by the LLM based on the information it has

ToDo:
- Store the pieces of successful plans in a vector db for later use and reduce generation costs
- Continue to improve text parsing to deal with more edge cases
- More post-processing
- Re-evaluate preconditions as a requirement for task execution

Installation:
- Backend
  - Set the environment variable `OPENAI_KEY` to your OpenAI api key
  - Install Dependencies
    - Run `pip install -r requirements.txt`
  - Run Application
    - `python src/main.py`
    - Enter the initial state
      - This is any information that you want the system to know before it begins planning
      - You may put "None" or nothing in this input
    - Describe your goal
      - Enter the goal that the system is planning to reach
      - Ex: "eat a ham sandwich"
    - Default capabilities
      - These are the tools that the planner may consider using when creating the plan
      - This defaults to "Linux terminal, internet access", you can just press enter to use these

- Frontend:
  - Go into the frontend directory
    - `cd src/frontend`
  - Start the frontend
    - `npm start`

- Credits:
  - DaemonIB
  - GPT-4
  - Bard