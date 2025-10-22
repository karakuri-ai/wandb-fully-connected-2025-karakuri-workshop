"""Config"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Config"""

    openai_api_key: SecretStr = Field(description="OpenAIのAPIキー")
    weave_project_name: str = Field(description="Weaveのプロジェクト名")
    wandb_api_key: SecretStr = Field(description="WandbのAPIキー")

    model_config = SettingsConfigDict(env_file=".env")
