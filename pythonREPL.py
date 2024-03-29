from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
import getpass
import os
os.environ["OPENAI_API_KEY"] = getpass.getpass()
tools = [PythonREPLTool()]
from langchain.agents import create_openai_functions_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(ChatOpenAI(temperature=0), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)#type: ignore
agent_executor.invoke({"input": "Draw a circle"})