import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.prompts import USER_INFO_SYSTEM_PROMPT_SECTION
from task.tools.base import BaseTool
from task.tools.memory._models import UserProfile
from task.tools.memory.user_info_extractor import UserInfoExtractor
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
            user_info_extractor: UserInfoExtractor | None = None,
            user_profile: UserProfile | None = None,
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self.user_info_extractor = user_info_extractor
        self.user_profile = user_profile
        self._tools_dict: dict[str, BaseTool] = {
            tool.name: tool
            for tool in tools
        }
        self.state = {
            TOOL_CALL_HISTORY_KEY: []
        }
        self._background_tasks: set[asyncio.Task] = set()

    async def handle_request(
            self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        api_key = request.api_key

        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )

        chunks = await client.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            stream=True,
            deployment_name=deployment_name,
        )

        tool_call_index_map = {}
        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.id:
                            tool_call_index_map[tool_call_delta.index] = tool_call_delta
                        else:
                            tool_call = tool_call_index_map[tool_call_delta.index]
                            if tool_call_delta.function:
                                argument_chunk = tool_call_delta.function.arguments or ''
                                tool_call.function.arguments += argument_chunk

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            custom_content=custom_content,
            tool_calls=[ToolCall.validate(tool_call) for tool_call in tool_call_index_map.values()]
        )

        if assistant_message.tool_calls:
            tasks = [
                self._process_tool_call(
                    tool_call=tool_call,
                    choice=choice,
                    api_key=api_key,
                    conversation_id=request.headers['x-conversation-id']
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)

            self.state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            return await self.handle_request(
                deployment_name=deployment_name,
                choice=choice,
                request=request,
                response=response
            )

        choice.set_state(self.state)

        if self.user_info_extractor and self.user_profile:
            latest_user_message = self._get_latest_user_message(request.messages)
            if latest_user_message:
                task = asyncio.create_task(
                    self._extract_user_info_safe(
                        api_key=api_key,
                        user_message=latest_user_message,
                        assistant_message=content,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        return assistant_message

    def _build_system_prompt(self) -> str:
        """
        Build the full system prompt including known user information.
        """
        prompt = self.system_prompt

        if self.user_profile and self.user_profile.info:
            lines = []
            for key, value in self.user_profile.info.items():
                readable_key = key.replace("_", " ").title()
                lines.append(f"- {readable_key}: {value}")
            user_info_text = "\n".join(lines)
            if user_info_text:
                prompt += USER_INFO_SYSTEM_PROMPT_SECTION.format(user_info=user_info_text)

        return prompt

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        unpacked_messages = unpack_messages(messages, self.state[TOOL_CALL_HISTORY_KEY])
        unpacked_messages.insert(
            0,
            {
                "role": Role.SYSTEM.value,
                "content": self._build_system_prompt(),
            }
        )

        print("\nHistory:")
        for msg in unpacked_messages:
            print(f"     {json.dumps(msg)}")

        print(f"{'-' * 100}\n")

        return unpacked_messages

    @staticmethod
    def _get_latest_user_message(messages: list[Message]) -> str | None:
        """Extract the content of the latest user message from the conversation."""
        for message in reversed(messages):
            if message.role == Role.USER:
                return message.content or ""
        return None

    async def _extract_user_info_safe(
            self,
            api_key: str,
            user_message: str,
            assistant_message: str,
    ) -> None:
        """
        Safely run user info extraction, catching any exceptions
        to avoid disrupting the main response flow.
        """
        try:
            await self.user_info_extractor.process_after_response(
                api_key=api_key,
                user_message=user_message,
                assistant_message=assistant_message,
                current_profile=self.user_profile,
            )
        except Exception as e:
            print(f"[UserInfoExtractor] Background extraction failed: {e}")

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[
        str, Any]:
        tool_name = tool_call.function.name
        stage = StageProcessor.open_stage(
            choice,
            tool_name
        )

        tool = self._tools_dict[tool_name]

        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r")
            stage.append_content("## Response: \n")

        tool_message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id
            )
        )

        StageProcessor.close_stage_safely(stage)

        return tool_message.dict(exclude_none=True)
