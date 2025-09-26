import pandas as pd
try:
    from db_connect import engine
    except ImportError as e:
    engine = None

def save_feedback(input_data: dict, prediction: float, feedback_text: str = None):
    """
    Guarda el feedback del usuario en la base de datos
    """
    if engine is None:
        raise Exception("Database engine not available - check db_connect.py")

    try:
        df = pd.DataFrame([{
            "input_data": str(input_data),
            "prediction": prediction,
            "feedback": feedback_text,
            "timestamp": pd.Timestamp.now()
        }])

        df.to_sql("feedback", engine, if_exists="append", index=False)
        } registro(s)")

    except Exception as e:
        raise e