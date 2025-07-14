// Initialize Three.js scene
let scene, camera, renderer, controls;
let bodies = [];
let trails = [];

function initScene() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, 
        document.getElementById('simulation-container').clientWidth / 
        document.getElementById('simulation-container').clientHeight, 
        0.1, 1000);
    camera.position.z = 15;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
        document.getElementById('simulation-container').clientWidth,
        document.getElementById('simulation-container').clientHeight
    );
    document.getElementById('simulation-container').appendChild(renderer.domElement);
    
    // Add controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
    
    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = document.getElementById('simulation-container').clientWidth / 
                        document.getElementById('simulation-container').clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(
            document.getElementById('simulation-container').clientWidth,
            document.getElementById('simulation-container').clientHeight
        );
    });
    
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Initialize body inputs
function initBodyInputs() {
    const container = document.getElementById('body-inputs');
    container.innerHTML = '';
    
    // Create 3 body inputs by default
    for (let i = 0; i < 3; i++) {
        addBodyInput(i + 1);
    }
}

function addBodyInput(bodyNumber) {
    const container = document.getElementById('body-inputs');
    
    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'body-input';
    bodyDiv.innerHTML = `
        <h3>Body ${bodyNumber}</h3>
        <div class="input-group">
            <label>Mass</label>
            <input type="number" class="mass" value="${bodyNumber === 1 ? 1.0 : 0.1}" step="0.1">
        </div>
        <div class="input-group">
            <label>Position (X, Y, Z)</label>
            <div class="position-inputs" style="display: flex; gap: 10px;">
                <input type="number" class="pos-x" value="${bodyNumber === 1 ? 0 : bodyNumber === 2 ? 1 : 0}" step="0.1">
                <input type="number" class="pos-y" value="${bodyNumber === 1 ? 0 : bodyNumber === 2 ? 0 : 1}" step="0.1">
                <input type="number" class="pos-z" value="0" step="0.1">
            </div>
        </div>
        <div class="input-group">
            <label>Velocity (X, Y, Z)</label>
            <div class="velocity-inputs" style="display: flex; gap: 10px;">
                <input type="number" class="vel-x" value="${bodyNumber === 1 ? 0 : bodyNumber === 2 ? 0 : -1}" step="0.1">
                <input type="number" class="vel-y" value="${bodyNumber === 1 ? 0 : bodyNumber === 2 ? 1 : 0}" step="0.1">
                <input type="number" class="vel-z" value="0" step="0.1">
            </div>
        </div>
    `;
    
    container.appendChild(bodyDiv);
}

// Handle prediction
async function predict() {
    const bodyInputs = document.querySelectorAll('.body-input');
    if (bodyInputs.length !== 3) {
        alert('Please enter exactly 3 bodies');
        return;
    }
    
    const masses = [];
    const positions = [];
    const velocities = [];
    
    bodyInputs.forEach(input => {
        masses.push(parseFloat(input.querySelector('.mass').value));
        
        const pos = [
            parseFloat(input.querySelector('.pos-x').value),
            parseFloat(input.querySelector('.pos-y').value),
            parseFloat(input.querySelector('.pos-z').value)
        ];
        positions.push(pos);
        
        const vel = [
            parseFloat(input.querySelector('.vel-x').value),
            parseFloat(input.querySelector('.vel-y').value),
            parseFloat(input.querySelector('.vel-z').value)
        ];
        velocities.push(vel);
    });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                masses: masses,
                positions: positions,
                velocities: velocities
            })
        });
        
        const data = await response.json();
        
        // Display prediction
        document.getElementById('prediction-result').textContent = 
            `Predicted Outcome: ${data.prediction}`;
        
        // Display confidence
        let confidenceHTML = '<h3>Confidence:</h3>';
        for (const [outcome, prob] of Object.entries(data.confidence)) {
            confidenceHTML += `<p>${outcome}: ${(prob * 100).toFixed(2)}%</p>`;
        }
        document.getElementById('confidence').innerHTML = confidenceHTML;
        
        // Visualize simulation
        visualizeSimulation(data.trajectories);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('prediction-result').textContent = 
            'Error: ' + error.message;
    }
}

// Visualize simulation
function visualizeSimulation(trajectories) {
    // Clear previous simulation
    while(scene.children.length > 0) { 
        scene.remove(scene.children[0]); 
    }
    
    // Add lighting back
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
    
    // Colors for bodies
    const colors = [0xff0000, 0x00ff00, 0x0000ff];
    bodies = [];
    trails = [];
    
    // Create bodies and trails
    for (let i = 0; i < 3; i++) {
        // Create body
        const geometry = new THREE.SphereGeometry(0.2, 32, 32);
        const material = new THREE.MeshPhongMaterial({ color: colors[i] });
        const body = new THREE.Mesh(geometry, material);
        scene.add(body);
        bodies.push(body);
        
        // Create trail
        const trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.LineBasicMaterial({ color: colors[i] });
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trail);
        trails.push(trail);
    }
    
    // Animate simulation
    let frame = 0;
    const maxFrames = trajectories[0].length;
    
    function animateSimulation() {
        if (frame >= maxFrames) return;
        
        for (let i = 0; i < 3; i++) {
            // Update body position
            const [x, y, z] = trajectories[i][frame];
            bodies[i].position.set(x, y, z);
            
            // Update trail
            const positions = [];
            for (let j = 0; j <= frame; j++) {
                const [tx, ty, tz] = trajectories[i][j];
                positions.push(tx, ty, tz);
            }
            
            trails[i].geometry.setAttribute(
                'position',
                new THREE.Float32BufferAttribute(positions, 3)
            );
        }
        
        frame++;
        requestAnimationFrame(animateSimulation);
    }
    
    animateSimulation();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initScene();
    initBodyInputs();
    
    // Add event listeners
    document.getElementById('add-body-btn').addEventListener('click', () => {
        const bodyCount = document.querySelectorAll('.body-input').length;
        if (bodyCount >= 3) {
            alert('Only 3 bodies are supported for the three-body problem');
            return;
        }
        addBodyInput(bodyCount + 1);
    });
    
    document.getElementById('predict-btn').addEventListener('click', predict);
});
