
## Saving Checkpoints with PostgresSaver

```python   
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, List
from typing_extensions import TypedDict
from psycopg import Connection

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

DB_URI = "postgresql://postgres:example@localhost:5432/postgres?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(DB_URI, **connection_kwargs)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="your_groq_api_key")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")
])
llm_model = prompt_template | llm

graph_builder = StateGraph(State)

def ChatNode(state: State) -> State:
    system_message = "You are an assistant"
    state["messages"] = llm_model.invoke({"system_message": system_message, "messages": state["messages"]})
    return state

graph_builder.add_node("chatnode", ChatNode)
graph_builder.add_edge(START, "chatnode")
graph_builder.add_edge("chatnode", END)
graph = graph_builder.compile(checkpointer=checkpointer)

# Example interaction
config = {"configurable": {"thread_id": "123"}}
input_state = {"messages": ["My name is Sajith"]}
response_state = graph.invoke(input_state, config=config)
for message in response_state["messages"]:
    message.pretty_print()

# Continue the conversation
input_state = {"messages": ["Who am I?"]}
response_state = graph.invoke(input_state, config=config)
for message in response_state["messages"]:
    message.pretty_print()

conn.close()
```

# Saving Memory with PostgreStore

```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, List, Optional
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from pydantic import BaseModel
from typing_extensions import TypedDict

DB_URI = "postgresql://postgres:example@localhost:5432/postgres?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(DB_URI, **connection_kwargs)
store = PostgresStore(conn)
store.setup()

user_id = "1"
namespace_for_memory = (user_id, "memories")

class Profile(BaseModel):
    name: Optional[str]
    age: Optional[int]
    profession: Optional[str]
    hobby: Optional[List[str] | str]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="your_groq_api_key")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")
])
llm_model = prompt_template | llm
structured_llm = prompt_template | llm.with_structured_output(Profile)

graph_builder = StateGraph(State)

def GetProfileNode(state: State, store: BaseStore) -> State:
    system_message = """You are an expert in finding profile information from user-AI communication: {context}.
    Don't hallucinate; return None for unknown data."""
    result = structured_llm.invoke({"system_message": system_message.format(context=state["messages"]), "messages": [{"role": "user", "content": "get my profile info"}]})
    
    profile = store.get(namespace_for_memory, key="profile")
    profile = profile.value if profile else {'name': None, 'age': None, 'profession': None, 'hobby': []}
    
    profile['name'] = result.name if result.name else profile['name']
    profile['age'] = result.age if result.age else profile['age']
    profile['profession'] = result.profession if result.profession else profile['profession']
    if result.hobby:
        profile['hobby'] += [result.hobby] if isinstance(result.hobby, str) else result.hobby
        
    store.put(namespace_for_memory, "profile", profile)
    return state

def ChatNode(state: State, store: BaseStore) -> State:
    system_message = "You are an assistant. Use context from previous communication: {context}"
    memories = store.search(namespace_for_memory, query=state["messages"][-1].content, limit=3)
    context = "".join(str(memory.value) for memory in memories)
    state["messages"] = llm_model.invoke({"system_message": system_message.format(context=context), "messages": state["messages"]})
    return state

graph_builder.add_node("ChatNode", ChatNode)
graph_builder.add_node("GetProfileNode", GetProfileNode)
graph_builder.add_edge(START, "ChatNode")
graph_builder.add_edge("ChatNode", "GetProfileNode")
graph_builder.add_edge("GetProfileNode", END)

graph = graph_builder.compile(checkpointer=MemorySaver(), store=store)

# Example interaction
config = {"configurable": {"thread_id": "321"}}
input_state = {"messages": ["My name is Sajith"]}
response_state = graph.invoke(input_state, config=config)
for message in response_state["messages"]:
    message.pretty_print()

```

## Vector Search with PostgresStore

```python
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

DB_URI = "postgresql://postgres:example@localhost:5432/postgres?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(DB_URI, **connection_kwargs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")
store = PostgresStore(conn, index={"embed": embeddings, "dims": 768})
store.setup()

user_id = "1"
namespace_for_memory = (user_id, "memories")

# Store data with embeddings
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"food_preference": "I love Italian cuisine", "context": "Discussing dinner plans"},
    index=["food_preference"]
)

# Search for relevant memories
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3
)
for memory in memories:
    print(memory)

```

Refrence
https://medium.com/@sajith_k/using-postgresql-with-langgraph-for-state-management-and-vector-storage-df4ca9d9b89e