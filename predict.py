import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# 1. Define the network architecture (must match training)
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, action_size)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# 2. Initialize the agent
state_size = 21  # Number of features (3 masses + 3*3 positions + 3*3 velocities)
action_size = 3   # Number of possible outcomes
loaded_agent = DQN(state_size, action_size).to('cpu')
loaded_agent.load_state_dict(torch.load('models/three_body_rl_agent_pytorch.pth', map_location='cpu'))
loaded_agent.eval()

# 3. Load preprocessing objects
# Create preprocessors directory if it doesn't exist
os.makedirs('preprocessors', exist_ok=True)

scaler_params = np.load('preprocessors/scaler_params.npz')
label_encoder_classes = np.load('preprocessors/label_encoder_classes.npy', allow_pickle=True)

scaler = StandardScaler()
scaler.mean_ = scaler_params['mean']
scaler.scale_ = scaler_params['scale']

label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# 4. Prediction function
def predict_three_body(masses, positions, velocities):
    """
    Predict outcome for new three-body system
    
    Parameters:
        masses: [m1, m2, m3]
        positions: [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
        velocities: [[vx1,vy1,vz1], [vx2,vy2,vz2], [vx3,vy3,vz3]]
    
    Returns: (predicted_class, confidence_dict)
    """
    # Flatten and scale input
    flat_pos = np.array(positions).flatten()
    flat_vel = np.array(velocities).flatten()
    features = np.concatenate([masses, flat_pos, flat_vel])
    scaled_features = scaler.transform([features])[0]
    
    # Get prediction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0)
        q_values = loaded_agent(input_tensor)
        probs = torch.softmax(q_values, dim=1).numpy()[0]
    
    # Return human-readable results
    pred_class = label_encoder.inverse_transform([q_values.argmax().item()])[0]
    confidence = dict(zip(label_encoder.classes_, probs))
    
    return pred_class, confidence



# 5. Example usage
if __name__ == "__main__":
    # Example system (same format as training data)
    masses = [1.0, 0.1, 0.1]  # Central body is heavy, others are light
    positions = [[0, 0, 0],[1, 0, 0],[0, 1, 0]]
    velocities = [[0, 0, 0],[0, 1, 0],[-1, 0, 0]]

    # Make prediction
    prediction, confidence = predict_three_body(
        masses,
        positions,
        velocities
    )
    
    print("\nThree-Body System Prediction:")
    print(f"Predicted outcome: {prediction}")
    print("Confidence:")
    for outcome, prob in confidence.items():
        print(f"  {outcome}: {prob:.2%}")