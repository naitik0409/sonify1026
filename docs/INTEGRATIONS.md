# Integrations

Give your agents, assistants and apps an expressive Indian female voice.
Sonify 1026 ships a **Model Context Protocol (MCP)** server plus drop-in
helpers for LangChain and LlamaIndex.

- [MCP server](#mcp-server) — Claude Desktop, Cursor, Windsurf, any MCP client
- [LangChain](#langchain)
- [LlamaIndex](#llamaindex)
- [Generic function calling](#generic-function-calling)
- [Tool reference](#tool-reference)

Install the optional MCP extra:

```bash
pip install "sonify1026[mcp]"     # or: pip install "snfy[mcp]"
```

---

## MCP server

The server exposes two tools — `synthesize` and `emotions` — and speaks
over stdio. Run it directly:

```bash
snfy-mcp                          # console script (installed with the extra)
python -m snfy.mcp_server         # equivalent
```

Audio files are written to `$SNFY_OUTPUT_DIR`, falling back to a `snfy/`
folder in the system temp directory. Each render gets a unique, timestamped
filename so nothing is overwritten.

### Claude Desktop

Open **Settings → Developer → Edit Config** (or the file directly) and add a
server entry. Config locations: `%APPDATA%\Claude\claude_desktop_config.json`
(Windows), `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS).

```json
{
  "mcpServers": {
    "sonify1026": {
      "command": "python",
      "args": ["-m", "snfy.mcp_server"]
    }
  }
}
```

The same snippet is in [`examples/claude_desktop_config.json`](../examples/claude_desktop_config.json).
Restart Claude Desktop, then ask it to "say *Yaar, this works* warmly".

> If `python` is not the interpreter that has Sonify installed, use its full
> path, or the `snfy-mcp` executable in `"command"` with `"args": []`.

### Cursor / Windsurf / other MCP clients

Any client that accepts an MCP `command`/`args` pair works the same way. Point
it at `python -m snfy.mcp_server` (stdio transport).

### Programmatic MCP clients

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["-m", "snfy.mcp_server"])

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "synthesize", {"text": "Namaste!", "emotion": "warm"}
            )
            print(result)

asyncio.run(main())
```

---

## LangChain

```python
from langchain_core.tools import tool
from snfy.mcp_server import synthesize_to_file

@tool
def speak(text: str, emotion: str = "neutral", language: str = "auto") -> str:
    """Speak text aloud with an expressive Indian female voice; returns the WAV path."""
    return synthesize_to_file(text, emotion=emotion, language=language)["path"]

# tools=[speak, ...]  ->  attach to any agent
```

Full example: [`examples/langchain_tts.py`](../examples/langchain_tts.py).

```bash
pip install langchain-core "sonify1026[mcp]"
python examples/langchain_tts.py
```

---

## LlamaIndex

```python
from llama_index.core.tools import FunctionTool
from snfy.mcp_server import synthesize_to_file

def speak(text: str, emotion: str = "neutral", language: str = "auto") -> str:
    """Speak text aloud with an expressive Indian female voice; returns the WAV path."""
    return synthesize_to_file(text, emotion=emotion, language=language)["path"]

speak_tool = FunctionTool.from_defaults(fn=speak)
```

Full example: [`examples/llamaindex_tts.py`](../examples/llamaindex_tts.py).

```bash
pip install llama-index-core "sonify1026[mcp]"
python examples/llamaindex_tts.py
```

---

## Generic function calling

For OpenAI-style function calling (or any framework), declare a function that
calls `synthesize_to_file` and returns the path:

```json
{
  "name": "speak",
  "description": "Speak text aloud with an expressive Indian female voice. Returns the path to a WAV file.",
  "parameters": {
    "type": "object",
    "properties": {
      "text": { "type": "string", "description": "The words to speak." },
      "emotion": {
        "type": "string",
        "description": "A Sonify emotion preset, e.g. warm, happy, calm, whisper.",
        "default": "neutral"
      },
      "language": {
        "type": "string",
        "enum": ["auto", "english", "hindi", "us"],
        "default": "auto"
      }
    },
    "required": ["text"]
  }
}
```

```python
from snfy.mcp_server import synthesize_to_file

def speak(text, emotion="neutral", language="auto"):
    return synthesize_to_file(text, emotion=emotion, language=language)["path"]
```

---

## Tool reference

### `synthesize`

| Argument | Type | Default | Notes |
|---|---|---|---|
| `text` | string | — | English, Hindi or Hinglish; required, non-empty |
| `emotion` | string | `"neutral"` | any of the 17 presets (`emotions` lists them) |
| `language` | string | `"auto"` | `auto`, `english`/`en-in`, `hindi`/`hi-in`, `us`/`en-us` |
| `speed` | float | `1.0` | `0.5`–`2.0` |
| `pitch` | float | `0.0` | `-20`–`+25` Hz |

Returns `{ "path", "duration", "sample_rate", "emotion", "language" }`.

### `emotions`

Returns the list of available emotion preset names.

The underlying helper is importable directly, with no MCP server required:

```python
from snfy.mcp_server import synthesize_to_file, list_emotions

info = synthesize_to_file("Take your time.", emotion="warm")
print(info["path"], info["duration"])
```

> Synthesis needs an internet connection (audio is served by Microsoft Edge
> TTS). See [Requirements and limits](../README.md#requirements-and-limits).
