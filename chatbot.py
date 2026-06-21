from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
app = graph.compile()

history = []
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": history})
    history = result["messages"]
    print("Bot:", history[-1].content)