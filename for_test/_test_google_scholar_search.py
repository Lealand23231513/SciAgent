from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from global_var import _init
import logging
import os
import os
import logging

from google_scholar_search import get_google_scholar_search_tool
from pathlib import Path
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from pathlib import Path


def scholar_sarch_with_agent(user_input: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)
    tools = [
        get_google_scholar_search_tool(load_all_available_meta=True, download=False)
    ]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, #type: ignore
        tools=tools, 
        handle_parsing_errors=True
    )
    ans = agent_executor.invoke({"input": user_input})
    logger.info(ans)
    return ans["output"]


load_dotenv()
_init()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(Path(__file__).stem)
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
res = scholar_sarch_with_agent("Find some papers about AI")
print(res)
