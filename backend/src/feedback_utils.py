import pandas as pd

# Importar db_connect desde el mismo directorio
try:
    from db_connect import engine
    print("✅ Database engine imported successfully")
except ImportError as e:
    print(f"Error importing db_connect: {e}")
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
        print(f"Feedback guardado exitosamente: {len(df)} registro(s)")
        
    except Exception as e:
        print(f"Error al guardar feedback: {e}")
        raise e