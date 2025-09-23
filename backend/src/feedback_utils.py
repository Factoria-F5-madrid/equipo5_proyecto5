import pandas as pd
from db_connection import engine

def save_feedback(input_data: dict, prediction: float, feedback_text: str = None):
    df = pd.DataFrame([{
        "input_data": str(input_data),
        "prediction": prediction,
        "feedback": feedback_text
    }])
    df.to_sql("feedback", engine, if_exists="append", index=False)
