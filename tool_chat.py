from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama


@tool
def add_numbers(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회합니다."""
    weather_data = {
        "서울": "맑음, 15도",
        "Seoul": "맑음, 15도",  # ← 추가
        "seoul": "맑음, 15도",  # ← 추가 (소문자도)
        "부산": "흐림, 18도",
        "Busan": "흐림, 18도",  # ← 추가
        "제주": "비, 20도",
        "Jeju": "비, 20도",  # ← 추가
    }
    return weather_data.get(city, "날씨 정보 없음")


tools = [add_numbers, get_weather]

llm = ChatOllama(model="qwen3:14b")
llm_with_tools = llm.bind_tools(tools)
SYSTEM_PROMPT = "You must always respond in Korean only. 반드시 한국어로만 답변하세요. 영어로 절대 답변하지 마세요."


def call_llm(state: MessagesState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


memory = MemorySaver()

graph = StateGraph(MessagesState)
graph.add_node(call_llm)
graph.add_node(ToolNode(tools))

graph.add_edge(START, "call_llm")
graph.add_conditional_edges("call_llm", tools_condition)
graph.add_edge("tools", "call_llm")

graph = graph.compile(checkpointer=memory)

CONFIG = {"configurable": {"thread_id": "tool-session"}}

# 테스트 1 - Tool 사용
result = graph.invoke(
    {"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]},
    config=CONFIG
)
print(result["messages"][-1].content)

# 테스트 2 - Tool 사용
result = graph.invoke(
    {"messages": [{"role": "user", "content": "253 + 847 계산해줘"}]},
    config=CONFIG
)
print(result["messages"][-1].content)

# 테스트 3 - Tool 불필요
result = graph.invoke(
    {"messages": [{"role": "user", "content": "안녕!"}]},
    config=CONFIG
)
print(result["messages"][-1].content)
