from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="lfm2.5-thinking:1.2b")


def classify(state: MessagesState):
    last_message = state["messages"][-1].content
    response = llm.invoke([
        {"role": "system", "content": "사용자 메시지가 긍정이면 'positive', 부정이면 'negative'만 답해."},
        {"role": "user", "content": last_message}
    ])
    return {"messages": [response]}


def positive_response(state: MessagesState):
    response = llm.invoke([
        {"role": "system", "content": "긍정적인 사용자에게 신나게 응답해줘!"},
        *state["messages"]
    ])
    return {"messages": [response]}


def negative_response(state: MessagesState):
    response = llm.invoke([
        {"role": "system", "content": "힘들어하는 사용자를 따뜻하게 위로해줘."},
        *state["messages"]
    ])
    return {"messages": [response]}


def router(state: MessagesState):
    last_content = state["messages"][-1].content
    if "positive" in last_content:
        return "positive_response"
    else:
        return "negative_response"


graph = StateGraph(MessagesState)
graph.add_node(classify)
graph.add_node(positive_response)
graph.add_node(negative_response)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", router)
graph.add_edge("positive_response", END)
graph.add_edge("negative_response", END)

graph = graph.compile()

result = graph.invoke({
    "messages": [{"role": "user", "content": "오늘 주식으로 100만원 벌었어요!"}]
})

print(result["messages"][-1].content)
