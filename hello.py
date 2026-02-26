from langgraph.graph import StateGraph, MessagesState, START, END

# 1. Node 정의 - LLM 역할을 하는 함수
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello dongho"}]}

# 2. Graph 빌더 생성 (Spring의 @Configuration 클래스 같은 것)
graph = StateGraph(MessagesState)

# 3. 노드 등록 (함수 이름이 자동으로 노드 ID가 됨 -> "mock_llm")
graph.add_node(mock_llm)

# 4. 엣지 연결
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)

graph = graph.compile()

# 6. 실행 (Spring의 jobLauncher.run() 같은 것)
result = graph.invoke({
    "message": [{"role": "user", "content": "hi"}]
})

print(result["messages"][0].content)
