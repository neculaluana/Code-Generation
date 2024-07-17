import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


def read_api_key(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()


api_key_path = "serpapi"
serpapi_api_key = read_api_key(api_key_path)

llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)


agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


response = agent_executor.invoke({"input": "Find the 5-digit prime numbers."})
print(response)
