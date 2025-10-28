from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration class that loads and validates environment variables for the application.
    """

    # --- Environment loading ---
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI Configuration ---
    OPENAI_API_KEY: str = Field(description="API key for OpenAI service authentication.")

    TOOL_REGISTRY_URL: str = Field(description="URL of the tool registry global MCP server")


    @field_validator("OPENAI_API_KEY", "TOOL_REGISTRY_URL")
    @classmethod
    def check_not_empty(cls, value: str, info) -> str:
        """
        Validator to ensure that required fields are not empty.
        Logs an error and raises ValueError if a required field is missing or blank.
        """
        if not value or value.strip() == "":
            field_name = info.field_name or "unknown field"
            logger.error(f"{field_name} cannot be empty.")
            raise ValueError(f"{field_name} cannot be empty.")
        return value


try:
    settings = Settings()
except Exception as e:
    logger.error(f"❌ Failed to load configuration: {e}")
    raise SystemExit(e)
