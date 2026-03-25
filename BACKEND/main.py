from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import inference
from pydantic import BaseModel

label = ["Negative","Positive"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_headers = ["*"],
    allow_methods = ["*"],
    allow_credentials = True,
    allow_origins = ["*"]
)


class input_format(BaseModel):
    input_text:str

@app.post("/sentiment")
def predict_sentiment(data:input_format):
    pred_class,conf = inference(data.input_text)
    conf = conf*100
    conf = round(conf,2)
    # print("pred_class",pred_class)
    return {
        "pred_class":label[pred_class],
        "conf":conf
    }