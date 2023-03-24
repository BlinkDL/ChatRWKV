from flask import Flask, request
from inference import Inference
app = Flask(__name__)
infer_server = Inference()

@app.route("/api/chat", methods=['GET'])
def rwkv_chat():
    context = request.args.get("text")
    temperature = float(request.args.get("temperature"))
    top_p = float(request.args.get("topP"))
    token_count = int(request.args.get("tokenCount"))
    fresence_penalty = float(request.args.get("presencePenalty"))
    count_penalty = float(request.args.get("countPenalty"))
    return infer_server.predict(context,temperature,top_p,token_count,fresence_penalty,count_penalty)

@app.route("/api/write", methods=['GET'])
def rwkv_write():
    context = request.args.get("text")
    temperature = float(request.args.get("temperature"))
    top_p = float(request.args.get("topP"))
    token_count = int(request.args.get("tokenCount"))
    fresence_penalty = float(request.args.get("presencePenalty"))
    count_penalty = float(request.args.get("countPenalty"))
    return infer_server.predict(context,temperature,top_p,token_count,fresence_penalty,count_penalty)
