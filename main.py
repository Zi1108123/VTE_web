from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import json

# 载入模型与变量
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("variables.json", "r") as f:
    feature_names = json.load(f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": feature_names})

@app.post("/predict")
async def predict(request: Request):
    form = await request.form()
    
    # 将输入转为 float
    values = [float(form[f]) for f in feature_names]
    
    # 预测
    X = np.array(values).reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": feature_names,
            "result": round(float(prob), 4)
        }
    )
