# Postgres Vector Search

```python
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from langchain.embeddings.ollama import OllamaEmbeddings
import uuid

DB_URI = "postgresql://postgres:1234@localhost:5436/langchain"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(DB_URI, **connection_kwargs)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")
embeddings =OllamaEmbeddings(model="nomic-embed-text:latest")
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

## We can store this with React Agent

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.utils.config import get_store 
from langmem import (
    # Lets agent create, update, and delete memories 
    create_manage_memory_tool,
)


def prompt(state):
    """Prepare the messages for the LLM."""
    # Get store from configured contextvar; 
    store = get_store() # Same as that provided to `create_react_agent`
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]



checkpointer = MemorySaver() # Checkpoint graph state 

agent = create_react_agent( 
    llm,
    prompt=prompt,
    tools=[ # Add memory tools 
        # The agent can call "manage_memory" to
        # create, update, and delete memories by ID
        # Namespaces add scope to memories. To
        # scope memories per-user, do ("memories", "{user_id}"): 
        create_manage_memory_tool(namespace=("memories",)),
    ],
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    # And the graph "state" will be checkpointed after each node
    # completes executing for tracking the chat history and durable execution
    checkpointer=checkpointer, 
)

#invoke model

config = {"configurable": {"thread_id": "thread-a"}}

# Use the agent. The agent hasn't saved any memories,
# so it doesn't know about us
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,
)
print(response["messages"][-1].content)
# Output: "I don't seem to have any stored memories about your display mode preferences..."

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark. Remember that."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with
    # the same thread_id
    config=config,
)

# New thread = new conversation!
new_config = {"configurable": {"thread_id": "thread-b"}}
# The agent will only be able to recall
# whatever it explicitly saved using the manage_memories tool
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=new_config,
)
print(response["messages"][-1].content)
# Output: config = {"configurable": {"thread_id": "thread-a"}}

# Use the agent. The agent hasn't saved any memories,
# so it doesn't know about us
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,
)
print(response["messages"][-1].content)
# Output: "I don't seem to have any stored memories about your display mode preferences..."

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark. Remember that."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with
    # the same thread_id
    config=config,
)

# New thread = new conversation!
new_config = {"configurable": {"thread_id": "thread-b"}}
# The agent will only be able to recall
# whatever it explicitly saved using the manage_memories tool
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=new_config,
)
print(response["messages"][-1].content)
# Output: I don't currently have information about your preferred display mode. Would you like to specify your preference (e.g., dark mode, light mode, system default) so I can remember it for future interactions?
I remember you! Based on my records, your preference is for a "dark" theme or setting. Would you like me to update or expand on this information? Feel free to share more about your preferences, and I can adjust my memory accordingly

````