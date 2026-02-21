"""
c1_chatbot.py
─────────────
Hindi Conversational Chatbot using LangChain + Ollama.
Features:
  - Persistent conversation memory (last N turns)
  - Tool: Wikipedia lookup (Hindi)
  - Tool: Weather (OpenWeatherMap, if API key configured)
  - Responds in Hindi

Usage:
    python section_c/c1_chatbot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.ollama_client import ollama_chat

from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
import requests
import wikipedia


# ── Tools ──────────────────────────────────────────────────────────────────────

def wikipedia_search(query: str) -> str:
    """Search Hindi Wikipedia for a query."""
    try:
        wikipedia.set_lang("hi")
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia से जानकारी नहीं मिली: {e}"


def weather_tool(city: str) -> str:
    """Get current weather for a city."""
    if not config.OPENWEATHER_API_KEY:
        return ollama_chat(f"{city} के मौसम के बारे में एक सामान्य जानकारी दें।")
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={config.OPENWEATHER_API_KEY}&units=metric&lang=hi"
        )
        data = requests.get(url, timeout=10).json()
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{city} में अभी: {desc}, तापमान {temp}°C"
    except Exception as e:
        return f"मौसम की जानकारी उपलब्ध नहीं: {e}"


TOOLS = [
    Tool(name="Wikipedia", func=wikipedia_search,
         description="हिन्दी Wikipedia पर कुछ खोजने के लिए उपयोग करें।"),
    Tool(name="Weather", func=weather_tool,
         description="किसी शहर का मौसम जानने के लिए उपयोग करें। Input: city name."),
]


# ── Agent ──────────────────────────────────────────────────────────────────────

def build_agent() -> AgentExecutor:
    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL,
        temperature=0.4,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=True,
    )
    try:
        prompt = hub.pull("hwchase17/react-chat")
    except Exception:
        # Fallback if hub unavailable
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "तुम एक उपयोगी हिन्दी chatbot हो। हमेशा हिन्दी में जवाब दो।\n"
            "Chat History: {chat_history}\n"
            "Question: {input}\n"
            "Thought: {agent_scratchpad}"
        )
    agent = create_react_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def simple_chat_loop():
    """Simple fallback using direct Ollama calls with manual memory."""
    history: list[dict] = []
    print("🤖 हिन्दी चैटबॉट शुरू हो गया है। बाहर निकलने के लिए 'quit' टाइप करें।\n")
    SYSTEM = (
        "तुम एक उपयोगी हिन्दी भाषा सहायक हो। "
        "हमेशा हिन्दी में संक्षिप्त और स्पष्ट जवाब दो।"
    )
    import requests as req, json

    while True:
        user_input = input("आप: ").strip()
        if user_input.lower() in ("quit", "exit", "बाहर"):
            print("नमस्ते! 🙏")
            break

        history.append({"role": "user", "content": user_input})
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM}] + history[-12:],
            "stream": False,
        }
        resp = req.post(config.OLLAMA_CHAT_ENDPOINT, json=payload, timeout=60)
        reply = resp.json().get("message", {}).get("content", "")
        history.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")


def main():
    print("=" * 50)
    print("   हिन्दी AI चैटबॉट (Powered by Ollama qwen2:0.5b)")
    print("=" * 50)
    try:
        agent = build_agent()
        print("✅ LangChain Agent ready. Type 'quit' to exit.\n")
        while True:
            user = input("आप: ").strip()
            if user.lower() in ("quit", "exit"):
                break
            result = agent.invoke({"input": user})
            print(f"Bot: {result['output']}\n")
    except Exception as e:
        print(f"⚠  LangChain agent failed ({e}), using simple mode.")
        simple_chat_loop()


if __name__ == "__main__":
    main()
