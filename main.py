from flask import Flask, request
from ollama import Ollama

oll = Ollama()
app = Flask(__name__)


# curl -X POST -H 'Content-Type: application/json' localhost:8080 -d '{"query":"hello"}'
@app.route('/', methods=["POST"])
def index():
  return {
    "response": oll.generateResp(request.json.get("query"))
  }
  
app.run(host='0.0.0.0', port=8080)
