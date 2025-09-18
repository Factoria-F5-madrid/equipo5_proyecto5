from fastapi import FastAPI, UploadFile, File
import pandas as pd
from ods_pipeline import process_ods

app = FastAPI(title="ODS_API")

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    # Lee CSV recibido
    df = pd.read_csv(file.file)
    
    # Llama al pipeline de cálculo de ODS
    df_processed = process_ods(df)
    
    # Guarda CSV resultante en data/
    output_path = "data/ods_features.csv"
    df_processed.to_csv(output_path, index=False)
    
    return {"message": "Archivo procesado", "output_file": output_path}
