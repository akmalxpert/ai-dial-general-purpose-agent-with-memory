import json

from aidial_client import AsyncDial

from task.tools.memory._models import UserProfile
from task.tools.memory.user_profile_store import UserProfileStore
from task.prompts import USER_INFO_CHECK_PROMPT, USER_INFO_UPDATE_PROMPT


class UserInfoExtractor:
    """
    Extracts and updates user PII/personal details from conversations.

    Pipeline (runs after the final assistant response):
    1. Check: Use a mini/nano model to detect if the user message or assistant
       message contains any new user information (returns true/false).
    2. Update: If new info is detected, call the update chain with the existing
       profile + messages, and ask the model to produce an updated profile.
    3. Save: Persist the updated profile to storage.

    This two-step approach is an optimization — only ~1% of real conversations
    reveal new user information, so the expensive update step is rarely needed.
    """

    def __init__(
            self,
            endpoint: str,
            mini_deployment_name: str,
            user_profile_store: UserProfileStore,
    ):
        self.endpoint = endpoint
        self.mini_deployment_name = mini_deployment_name
        self.user_profile_store = user_profile_store

    async def process_after_response(
            self,
            api_key: str,
            user_message: str,
            assistant_message: str,
            current_profile: UserProfile,
    ) -> None:
        """
        Main entry point: check for new user info and update profile if needed.

        Called after the final assistant message is produced (no further tool calls).

        Args:
            api_key: DIAL API key for authentication
            user_message: The latest user message content
            assistant_message: The assistant's final response content
            current_profile: The currently stored user profile
        """
        # Step 1 — Fast check using mini model: does the conversation contain new user info?
        has_new_info = await self._check_for_new_info(
            api_key=api_key,
            user_message=user_message,
            assistant_message=assistant_message,
            current_profile=current_profile,
        )

        if not has_new_info:
            return

        # Step 2 — Update chain: extract new info and merge with existing profile
        updated_profile = await self._update_profile(
            api_key=api_key,
            user_message=user_message,
            assistant_message=assistant_message,
            current_profile=current_profile,
        )

        # Step 3 — Persist updated profile to storage
        await self.user_profile_store.save_profile(api_key, updated_profile)
        print(f"[UserInfoExtractor] Profile updated: {updated_profile.info}")

    async def _check_for_new_info(
            self,
            api_key: str,
            user_message: str,
            assistant_message: str,
            current_profile: UserProfile,
    ) -> bool:
        """
        Fast check using a mini/nano model to determine if the conversation
        contains any new user information not already in the profile.

        Returns True if new user info is detected, False otherwise.
        """
        profile_text = self.user_profile_store.format_profile_for_prompt(current_profile)
        if not profile_text:
            profile_text = "(No user information stored yet)"

        prompt = USER_INFO_CHECK_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
            current_profile=profile_text,
        )

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )

        try:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                deployment_name=self.mini_deployment_name,
                temperature=0.0,
                max_tokens=10,
                stream=False,
            )

            result_text = response.choices[0].message.content.strip().lower()
            print(f"[UserInfoExtractor] Check result: '{result_text}'")
            return result_text == "true"
        except Exception as e:
            print(f"[UserInfoExtractor] Check failed: {e}")
            return False

    async def _update_profile(
            self,
            api_key: str,
            user_message: str,
            assistant_message: str,
            current_profile: UserProfile,
    ) -> UserProfile:
        """
        Update the user profile by calling the mini model with the existing profile
        and the latest conversation messages.

        The model returns an updated JSON object with all user information.
        """
        current_info_json = json.dumps(current_profile.info, indent=2) if current_profile.info else "{}"

        prompt = USER_INFO_UPDATE_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message,
            current_profile_json=current_info_json,
        )

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )

        try:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                deployment_name=self.mini_deployment_name,
                temperature=0.0,
                stream=False,
            )

            result_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                result_text = "\n".join(lines)

            updated_info = json.loads(result_text)

            if not isinstance(updated_info, dict):
                print(f"[UserInfoExtractor] Update returned non-dict: {type(updated_info)}")
                return current_profile

            # Ensure all values are strings
            cleaned_info = {str(k): str(v) for k, v in updated_info.items() if v}

            updated_profile = current_profile.model_copy()
            updated_profile.info = cleaned_info
            return updated_profile

        except Exception as e:
            print(f"[UserInfoExtractor] Update failed: {e}")
            return current_profile

