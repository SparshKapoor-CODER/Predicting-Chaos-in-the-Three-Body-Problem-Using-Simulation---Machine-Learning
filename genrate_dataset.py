import numpy as np
import csv
from tqdm import tqdm
from RK4 import ThreeBodySimulator3D  # assumes RK4 simulator is in RK4.py
import pandas as pd
from edit import color_rows_based_on_column_v


# -------- Labeling Functions --------

def is_escape(positions, threshold=10.0):  # Increased threshold
    """Check if any body escapes the system."""
    for frame in positions:
        for pos in frame:
            if np.linalg.norm(pos) > threshold:
                return True
    return False

def is_collision(positions, threshold=0.1):
    """Optimized collision check"""
    for frame in positions:
        d01 = np.linalg.norm(frame[0]-frame[1])
        d02 = np.linalg.norm(frame[0]-frame[2])
        d12 = np.linalg.norm(frame[1]-frame[2])
        if d01 < threshold or d02 < threshold or d12 < threshold:
            return True
    return False

def get_label(positions):
    if is_collision(positions):
        return "collision"
    elif is_escape(positions):
        return "escape"
    return "stable"  # Only if neither condition met

# -------- Simulation & Dataset Generation --------

def simulate_one():
    # Random masses
    masses = np.round(np.random.uniform(0.5, 2.0, size=3), 3)

    # Random positions in a cube [-1, 1]
    positions = np.random.uniform(-1, 1, size=(3, 3))

    # Random velocities [-0.5, 0.5]
    velocities = np.random.uniform(-0.5, 0.5, size=(3, 3))

    # Run simulation
    sim = ThreeBodySimulator3D(masses, positions, velocities, dt=0.005)
    sim.run(steps=3000)
    traj = sim.get_trajectories()

    label = get_label(traj)

    flat_input = list(masses) + positions.flatten().tolist() + velocities.flatten().tolist()
    return flat_input + [label]

def generate_dataset(N=1000, output_file='three_body_dataset.csv'):
    print(f"Generating {N} labeled simulations...")

    # Define CSV header
    header = (
        ['m1', 'm2', 'm3'] +
        [f'p{i}_{axis}' for i in range(1, 4) for axis in 'xyz'] +
        [f'v{i}_{axis}' for i in range(1, 4) for axis in 'xyz'] +
        ['label']
    )

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for _ in tqdm(range(N)):
            row = simulate_one()
            writer.writerow(row)

    print(f"✅ Dataset saved to: {output_file}")


def csv_to_xlsx(csv_file, excel_file=None):
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Determine output filename if not provided
    if excel_file is None:
        if csv_file.endswith('.csv'):
            excel_file = csv_file[:-4] + '.xlsx'
        else:
            excel_file = csv_file + '.xlsx'
    
    # Write to Excel file
    df.to_excel(excel_file, index=False)
    
    print(f"✅ Successfully converted '{csv_file}' to '{excel_file}'")

# -------- Entry Point --------
if __name__ == '__main__':
    generate_dataset(N=20000)  # change to 1000 or more later
    csv_to_xlsx("three_body_dataset.csv","three_body_dataset.xlsx")

    input_file = "three_body_dataset.xlsx"  # Change to your input file path
    output_file = "three_body_dataset.xlsx"  # Change to your desired output path
    words = ["collision", "stable", "escape"]  # The three words to check for
    color_rows_based_on_column_v(input_file, output_file, words)
    print(f"✅ Changes done successfulley")