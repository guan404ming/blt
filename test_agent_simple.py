import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from blt.translators import LyricsTranslationAgent, LyricsTranslationAgentConfig

def test_simple():
    print("Initializing agent...")
    config = LyricsTranslationAgentConfig(
        model="qwen3:30b-a3b-instruct-2507-q4_K_M",
        ollama_base_url="http://localhost:11434",
    )
    
    agent = LyricsTranslationAgent(config=config)
    
    source_text = "我愛你"
    print(f"\nTranslating: {source_text}")
    
    try:
        result = agent.translate(
            source_lyrics=source_text,
            source_lang="cmn",
            target_lang="en-us"
        )
        
        print("\nTranslation Result:")
        for line in result.translated_lines:
            print(f"- {line}")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_simple()
