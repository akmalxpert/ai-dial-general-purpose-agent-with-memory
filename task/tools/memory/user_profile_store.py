import json
from datetime import datetime, UTC

from aidial_client import AsyncDial

from task.tools.memory._models import UserProfile


class UserProfileStore:
    """
    Manages user profile (PII and personal details) storage in DIAL bucket.

    Storage format: Single JSON file per user
    - File: {appdata_home}/__user-profile/profile.json
    - Caching: In-memory cache keyed by file path
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.cache: dict[str, UserProfile] = {}

    async def _get_profile_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the user profile file in DIAL bucket."""
        appdata_home = await dial_client.my_appdata_home()
        return f"files/{appdata_home}/__user-profile/profile.json"

    async def load_profile(self, api_key: str) -> UserProfile:
        """
        Load user profile from DIAL bucket.

        Returns cached version if available, otherwise loads from storage.
        Returns an empty profile if none exists.
        """
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        profile_file_path = await self._get_profile_file_path(dial_client)

        if profile_file_path in self.cache:
            return self.cache[profile_file_path]

        try:
            response = await dial_client.files.download(profile_file_path)
            content = (await response.aget_content()).decode('utf-8')
            data = json.loads(content)
            profile = UserProfile.model_validate(data)
        except Exception:
            profile = UserProfile(
                info={},
                updated_at=datetime.now(UTC)
            )

        self.cache[profile_file_path] = profile
        return profile

    async def save_profile(self, api_key: str, profile: UserProfile) -> None:
        """Save user profile to DIAL bucket and update cache."""
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        profile_file_path = await self._get_profile_file_path(dial_client)
        profile.updated_at = datetime.now(UTC)
        json_data = profile.model_dump_json()
        await dial_client.files.upload(url=profile_file_path, file=json_data.encode('utf-8'))
        self.cache[profile_file_path] = profile

    def format_profile_for_prompt(self, profile: UserProfile) -> str:
        """
        Format user profile as a readable string for inclusion in the system prompt.

        Returns an empty string if no user information is available.
        """
        if not profile.info:
            return ""

        lines = []
        for key, value in profile.info.items():
            readable_key = key.replace("_", " ").title()
            lines.append(f"- {readable_key}: {value}")

        return "\n".join(lines)

