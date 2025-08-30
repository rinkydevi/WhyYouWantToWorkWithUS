
# LangGraph Agent Setup
# Run: pip install -U langgraph "langchain[anthropic]"

import os
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def main():
    print("🌟 LangGraph Agent Demo")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  Setup required:")
        print("1. Get API key: https://console.anthropic.com/settings/keys")
        print("2. Set it: export ANTHROPIC_API_KEY='your-key-here'")
        print("\n📝 Demo without API (showing tool functionality):")
        print("User: what is the weather in sf")
        print("Tool result:", get_weather("sf"))
        return
    
    try:
        # Configure the LLM with specific parameters
        model = init_chat_model(
            "anthropic:claude-3-5-sonnet-20241022",
            temperature=0
        )
        
        # Add memory for multi-turn conversations
        checkpointer = InMemorySaver()
        
        # Create the agent with memory
        agent = create_react_agent(
            model=model,
            tools=[get_weather],
            prompt="You are a helpful assistant that can check weather information.",
            checkpointer=checkpointer
        )

        # Configuration for conversation thread
        config = {"configurable": {"thread_id": "1"}}
        
        print("🤖 Running LangGraph agent...")
        
        # First query
        result1 = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
            config
        )
        
        print("✅ First response:")
        for message in result1["messages"]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                print(f"   {message.type}: {message.content}")
        
        print("\n🔄 Testing memory with follow-up question...")
        
        # Follow-up query (tests memory)
        result2 = agent.invoke(
            {"messages": [{"role": "user", "content": "what about New York?"}]},
            config
        )
        
        print("✅ Follow-up response:")
        for message in result2["messages"]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                print(f"   {message.type}: {message.content}")
                
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install -U langgraph 'langchain[anthropic]'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
