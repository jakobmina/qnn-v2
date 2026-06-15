from flask import Flask, render_template, request, jsonify
from h7_qnn_hash import generate_metriplectic_hash
from utf8_qnn_poc import string_to_qnn_seed

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/hash", methods=["POST"])
def api_hash():
    data = request.json
    seed_val = data.get("seed", "")
    iterations = data.get("iterations", 3)
    
    try:
        iterations = int(iterations)
    except:
        iterations = 3

    if not seed_val:
        return jsonify({"error": "No seed provided"}), 400
        
    try:
        seed_n = int(seed_val)
    except ValueError:
        seed_n = string_to_qnn_seed(str(seed_val))
        
    try:
        result = generate_metriplectic_hash(seed_n, iterations)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
