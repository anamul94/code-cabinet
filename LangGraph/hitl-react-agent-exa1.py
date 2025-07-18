from langgraph.types import interrupt, Command
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. A tool that pauses for approval
def book_hotel(hotel_name: str) -> str:
    """Book a hotel."""
    # ⏸️ PAUSE here
    human_reply = interrupt(
        {
            "question": f"Approve booking '{hotel_name}'?",
            "type": "confirm"
        }
    )
    if human_reply["type"] == "accept":
        return f"✅ Booked {hotel_name}"
    else:
        return f"❌ Booking of {hotel_name} cancelled"

# 2. Create ReAct agent with memory
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[book_hotel],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "demo"}}

# 3. Run
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Book the Plaza"}]},
    config
):
    print(chunk)

# 4. Human sees the interrupt and responds
#    (In practice this is done by your UI or API)
for chunk in agent.stream(
    Command(resume={"type": "accept"}),   # or {"type": "reject"}
    config
):
    print(chunk)
