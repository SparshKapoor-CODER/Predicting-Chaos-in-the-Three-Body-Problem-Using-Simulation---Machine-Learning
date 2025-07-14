from flask import Flask, render_template, request, jsonify
import numpy as np
from predict import predict_three_body
from RK4 import ThreeBodySimulator3D

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    masses = data['masses']
    positions = data['positions']
    velocities = data['velocities']
    
    # Predict outcome
    prediction, confidence = predict_three_body(
        masses,
        positions,
        velocities
    )
    
    # Convert confidence values to native floats
    confidence = {k: float(v) for k, v in confidence.items()}
    
    # Run simulation
    sim = ThreeBodySimulator3D(masses, positions, velocities, dt=0.005)
    sim.run(steps=3000)
    traj = sim.get_trajectories()
    
    # Prepare trajectory data with native floats
    trajectories = []
    for body in range(3):
        body_traj = []
        for step in range(traj.shape[0]):
            # Convert each coordinate to native float
            body_traj.append([
                float(traj[step, body, 0]),
                float(traj[step, body, 1]),
                float(traj[step, body, 2])
            ])
        trajectories.append(body_traj)
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'trajectories': trajectories
    })

if __name__ == '__main__':
    app.run(debug=True)