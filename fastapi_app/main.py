from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

# Initialize FastAPI
app = FastAPI(title="E5 Text Classifier API")

# Downloading Tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

# Adress of our Triton Server
TRITON_URL = "triton:8001"
MODEL_NAME = "e5_classifier"

# Best threshold from our validation
OPTIMAL_THRESHOLD = 0.5325 

# Writing how our API should work
class InferenceRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        # 1. Data preparation (Tokenization)
        # max_length=64
        encoded = tokenizer(
            request.text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='np' # Returning Numpy arrays not Pytorch tensors
        )
        
        # Triton need INT64, as in config.pbtxt
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        # 2. Connecting Triton Server
        client = grpcclient.InferenceServerClient(url=TRITON_URL)
        
        # 3. Pckage our input (Inputs)
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # 4. Package our output (Outputs)
        outputs = [grpcclient.InferRequestedOutput("output")]

        # 5. Inference
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        
        # 6. post-processing
        # response.as_numpy() will return numpy array [1, 1], we need only 1 number
        logit = response.as_numpy("output")[0][0] 
        
        # Using sigmoid function
        probability = 1 / (1 + np.exp(-logit))
        
        # Using best threshold
        is_positive = bool(probability >= OPTIMAL_THRESHOLD)

        return {
            "text": request.text,
            "probability": float(probability),
            "prediction": 1 if is_positive else 0
        }

    except Exception as e:
        # If something gets wrong
        raise HTTPException(status_code=500, detail=str(e))