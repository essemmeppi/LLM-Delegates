from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from negotiation import NegotiationSystem
import os
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Initialize the negotiation system with your OpenAI API key
negotiation_system = NegotiationSystem(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/negotiate', methods=['POST'])
def negotiate():
    data = request.json
    
    shared_context = data.get('shared_context')
    cause_a = data.get('cause_a')
    cause_b = data.get('cause_b')

    if not all([shared_context, cause_a, cause_b]):
        return jsonify({"error": "Missing required parameters"}), 400

    def generate():
        try:
            for message in negotiation_system.run_negotiation_stream(shared_context, cause_a, cause_b):
                yield f"data: {json.dumps(message)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)