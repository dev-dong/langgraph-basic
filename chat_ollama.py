from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:14b", base_url="http://localhost:11434")

SYSTEM_PROMPT = "당신은 친절한 한국어 어시스턴트입니다. 항상 존댓말을 사용하세요."


def add_system_prompt(state: MessagesState):
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    return {"messages": [system_msg] + state["messages"]}


def call_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node(add_system_prompt)
graph.add_node(call_llm)
graph.add_edge(START, "add_system_prompt")
graph.add_edge("add_system_prompt", "call_llm")
graph.add_edge("call_llm", END)
graph = graph.compile()

result = graph.invoke({
    "messages": [{"role": "user", "content": "오늘 기분이 좋아요!"}]
})

print(result["messages"][-1].content)
