# analyze.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
from RK4 import ThreeBodySimulator3D  # Import your simulator

warnings.filterwarnings('ignore')

# Updated style configuration
try:
    plt.style.use('seaborn-v0_8')  # Newer matplotlib versions
except:
    plt.style.use('seaborn')  # Fallback for older versions
sns.set(style="whitegrid")

def main():
    # Load the dataset
    try:
        df = pd.read_csv('three_body_dataset.csv')
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
    except FileNotFoundError:
        print("Error: three_body_dataset.csv not found in current directory")
        return
    
    # Basic statistics
    print("\nLabel distribution:")
    print(df['label'].value_counts(normalize=True))
    
    # 1. Basic Visualizations
    def plot_label_distribution():
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
        plt.title('Distribution of Orbit Outcomes')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.show()
        plt.close()
    
    plot_label_distribution()
    
    # 2. Feature Correlations
    def plot_feature_correlations():
        numeric_cols = df.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
        plt.close()
    
    plot_feature_correlations()
    
    # 3. Trajectory Visualization (Sample Cases)
    def plot_sample_trajectories():
        # Load sample trajectories from simulations
        sample_cases = {}
        for label in ['stable', 'escape', 'collision']:
            if label in df['label'].unique():
                sample_cases[label] = df[df['label'] == label].iloc[0]
        
        if not sample_cases:
            print("Warning: No valid cases found for trajectory visualization")
            return
            
        fig = plt.figure(figsize=(6 * len(sample_cases), 6))
        
        for i, (outcome, row) in enumerate(sample_cases.items(), 1):
            # Reconstruct initial conditions
            masses = [row['m1'], row['m2'], row['m3']]
            positions = np.array([
                [row['p1_x'], row['p1_y'], row['p1_z']],
                [row['p2_x'], row['p2_y'], row['p2_z']],
                [row['p3_x'], row['p3_y'], row['p3_z']]
            ])
            velocities = np.array([
                [row['v1_x'], row['v1_y'], row['v1_z']],
                [row['v2_x'], row['v2_y'], row['v2_z']],
                [row['v3_x'], row['v3_y'], row['v3_z']]
            ])
            
            # Run simulation
            try:
                sim = ThreeBodySimulator3D(masses, positions, velocities, dt=0.005)
                sim.run(steps=3000)
                traj = sim.get_trajectories()
            except Exception as e:
                print(f"Error simulating {outcome} case: {str(e)}")
                continue
            
            # Plot
            ax = fig.add_subplot(1, len(sample_cases), i, projection='3d')
            colors = ['r', 'g', 'b']
            for body in range(3):
                ax.plot(traj[:, body, 0], traj[:, body, 1], traj[:, body, 2], 
                    color=colors[body], alpha=0.6, label=f'Body {body+1}')
                ax.scatter(traj[-1, body, 0], traj[-1, body, 1], traj[-1, body, 2], 
                        color=colors[body], s=100)
            
            ax.set_title(f'{outcome.capitalize()} Orbit')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    plot_sample_trajectories()
    
    # 4. Dimensionality Reduction
    def analyze_feature_space():
        # Prepare features (excluding labels)
        X = df.drop('label', axis=1)
        y = df['label']
        
        # PCA Analysis
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.7)
        plt.title('PCA of Initial Conditions')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # t-SNE Analysis
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', alpha=0.7)
        plt.title('t-SNE of Initial Conditions')
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return X_pca, X_tsne
    
    X_pca, X_tsne = analyze_feature_space()
    
    # 5. Clustering Analysis
    def perform_clustering(X_embedded):
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_embedded)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='tab10')
        plt.title('PCA with K-means Clusters')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='tab10')
        plt.title('t-SNE with K-means Clusters')
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Compare clusters with actual labels
        df['cluster'] = clusters
        cross_tab = pd.crosstab(df['cluster'], df['label'], normalize='index')
        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt=".2f")
        plt.title('Cluster vs Label Distribution')
        plt.show()
        plt.close()
    
    perform_clustering(X_tsne)
    
    

if __name__ == "__main__":
    main()