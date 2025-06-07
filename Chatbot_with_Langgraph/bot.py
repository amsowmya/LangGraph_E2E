from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from langchain_groq.chat_models import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api_key


class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")
        
    def call_tool(self):
        search = TavilySearchResults()
        tools = [search]
        self.tool_node = ToolNode(tools)
        self.llm_with_tool = self.llm.bind_tools(tools)
        
    def call_model(self, state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    def router_function(self, state:MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", 
                                       self.router_function,
                                       {
                                           "tools": "tools",
                                           END: END
                                       }
                                       )
        workflow.add_edge("tools", "agent")
        
        self.app = workflow.compile()
        return self.app 
    
if __name__ == "__main__":
    bot = Chatbot()
    workflow = bot()
    response = workflow.invoke({"messages": ["who is a current prime minister of USA?"]})
    print(response['messages'][-1].content)
        