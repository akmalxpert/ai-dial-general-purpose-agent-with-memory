import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Searches long-term memories about the user using semantic similarity. "
            "Use this to recall previously stored facts about the user such as preferences, personal info, "
            "goals, plans, or context. IMPORTANT: You MUST search memories at the START of EVERY new conversation "
            "to retrieve relevant user context before responding. Also search when the user asks anything that "
            "might relate to previously stored information. Returns the most relevant stored memories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results: list[MemoryData] = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        if not results:
            final_result = "No memories found."
        else:
            lines = []
            for r in results:
                line = f"- **{r.content}** (category: {r.category})"
                if r.topics:
                    line += f" [topics: {', '.join(r.topics)}]"
                lines.append(line)
            final_result = "\n".join(lines)

        tool_call_params.stage.append_content(final_result)
        return final_result
