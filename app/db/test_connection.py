from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.repositories.postgres import SessionLocal

def check_postgres_connection() -> None:
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1"))
        print("SUCCESS:", result.scalar())
    except OperationalError as e:
        print("FAILED:", repr(e))
        raise
    finally:
        db.close()

if __name__ == "__main__":
    check_postgres_connection()