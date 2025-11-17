{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from fastapi import FastAPI, Request\
from fastapi.templating import Jinja2Templates\
from fastapi.staticfiles import StaticFiles\
import numpy as np\
import pickle\
import json\
\
# \uc0\u36733 \u20837 \u27169 \u22411 \u19982 \u21464 \u37327 \
with open("model.pkl", "rb") as f:\
    model = pickle.load(f)\
\
with open("variables.json", "r") as f:\
    feature_names = json.load(f)\
\
app = FastAPI()\
templates = Jinja2Templates(directory="templates")\
\
@app.get("/")\
def index(request: Request):\
    return templates.TemplateResponse("index.html", \{"request": request, "features": feature_names\})\
\
@app.post("/predict")\
async def predict(request: Request):\
    form = await request.form()\
    \
    # \uc0\u23558 \u36755 \u20837 \u36716 \u20026  float\
    values = [float(form[f]) for f in feature_names]\
    \
    # \uc0\u39044 \u27979 \
    X = np.array(values).reshape(1, -1)\
    prob = model.predict_proba(X)[0, 1]\
    \
    return templates.TemplateResponse(\
        "index.html",\
        \{\
            "request": request,\
            "features": feature_names,\
            "result": round(float(prob), 4)\
        \}\
    )\
}