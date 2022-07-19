from fastapi import FastAPI, File, UploadFile
from train_sentiment import train_model
from sentiment_analysis import senti_test
import os
import uvicorn
app = FastAPI()


@app.post("/sentiment/train")
async def train_sentiment(text_are_name:str,label_are_name:str,dataframe: UploadFile = File(...)):
    file_name = os.path.basename(dataframe.filename)
    open(file_name, 'wb').write(dataframe.file.read())
    print(file_name)
    train_model(file_name,text_are_name,label_are_name)
    return {"Message": "Model training has started!"}

@app.post("/sentiment/testting")
async def test_sentiment(text:str):
    answer = senti_test(text)
    return {"Sentiment":answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")
