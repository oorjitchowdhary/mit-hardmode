"""Test display + LLM: ask Claude a question and show the response on the OLED."""
import time
from src.display.oled import OLEDDisplay
from src.llm.client import ClaudeClient

d = OLEDDisplay()
d.show_text("Asking Claude...")
print("Asking Claude a question...")

claude = ClaudeClient()
response = claude.ask("In one sentence, what is a Raspberry Pi?")

print(f"Response: {response}")
d.show_text(response)
time.sleep(8)
d.clear()
print("Done.")
