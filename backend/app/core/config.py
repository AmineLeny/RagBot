from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
  # Configuration JWT
  ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))
  SECRET_KEY: str = os.getenv('SECRET_KEY')
  ALGORITHM: str = os.getenv('ALGORITHM')

  # Supabase Configuration
  SUPABASE_URL: str = os.getenv('SUPABASE_URL')
  SUPABASE_KEY: str = os.getenv('SUPABASE_KEY')
  SUPABASE_BUCKET_NAME: str = os.getenv('SUPABASE_BUCKET_NAME')

  # Configuration de la base de données
  DATABASE_URL: str = "postgresql+psycopg2://postgres:amine123@localhost:5432/ragbot_db"
  
  # Rôles autorisés
  ALLOWED_ROLES: list[str] = ["admin", "user"]

  # Configuration d'OpenAI
  OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
  EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL')
  OPENAI_MODEL: str = os.getenv('OPENAI_MODEL')
  LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE'))
  LLM_MAX_TOKENS: int = int(os.getenv('LLM_MAX_TOKENS'))

settings = Settings()