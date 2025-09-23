import os
from sqlalchemy import create_engine

# Usa DATABASE_URL de Railway, o un fallback local
#DATABASE_URL = os.getenv("postgresql://admin:admin@localhost:5432/healthdb")
DATABASE_URL = database='healthdb', user='admin', password='admin'

engine = create_engine(DATABASE_URL)
