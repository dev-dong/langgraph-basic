from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:8b")
SYSTEM_PROMPT = "You are a friendly Korean-language assistant. Please always use polite/formal speech."


def call_llm(state: MessagesState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}


memory = MemorySaver()

graph = StateGraph(MessagesState)
graph.add_node(call_llm)
graph.add_edge(START, "call_llm")
graph.add_edge("call_llm", END)

graph = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-123"}}

# 첫 번째 대화
graph.invoke(
    {"messages": [{"role": "user", "content": "내 이름은 동호야"}]},
    config=config
)

# 두 번째 대화 (같은 thread_id!)
result = graph.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
    config=config
)

print(result["messages"][-1].content)
