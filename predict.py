import onnxruntime as rt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

iris = load_iris()
target_names = iris.target_names
features_name = iris.feature_names

class InputFeature(BaseModel):
    input_metric: list[float]

app = FastAPI()

# API endpoint
@app.post("/iris-prediction")
async def ml_endpoint(req: InputFeature):
    try:
        X_test = np.array([req.input_metric], dtype=np.float32)
        sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: X_test})
        return {"prediction": target_names[pred_onx[0][0]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=8080)