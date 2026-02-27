from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama


class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_name: str
    question_count: str


llm = ChatOllama(model="lfm2:24b")


def call_llm(state: MyState):
    user_name = state.get("user_name", "사용자")
    count = state.get("question_count", 0)

    messages = [
        SystemMessage(content=f"You must always respond in Korean. 사용자 이름은 {user_name}입니다. 이름을 불러주세요."),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "question_count": count + 1
    }


memory = MemorySaver()

graph = StateGraph(MyState)
graph.add_node(call_llm)
graph.add_edge(START, "call_llm")
graph.add_edge("call_llm", END)

graph = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "custom-state-session"}}

# 초기 State에 user_name 포함
result = graph.invoke(
    {
        "messages": [{"role": "user", "content": "안녕!"}],
        "user_name": "동호",
        "question_count": 0
    },
    config=config
)
print(result["messages"][-1].content)
print(f"질문 횟수: {result['question_count']}")

# 두 번째 질문
result = graph.invoke(
    {"messages": [{"role": "user", "content": "오늘 기분이 좋아!"}]},
    config=config
)
print(result["messages"][-1].content)
print(f"질문 횟수: {result['question_count']}")
