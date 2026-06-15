document.getElementById('btn-hash').addEventListener('click', async () => {
    const seed = document.getElementById('seed').value;
    const iterations = document.getElementById('iterations').value;

    if (!seed) return;

    // UI States
    const loader = document.getElementById('loader');
    const resultsPanel = document.getElementById('results');
    
    loader.classList.remove('hidden');
    resultsPanel.classList.add('hidden');

    try {
        const response = await fetch('/api/hash', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed, iterations })
        });

        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            loader.classList.add('hidden');
            return;
        }

        // Populate metrics
        document.getElementById('out-psi').innerText = data.hash.psi.toFixed(6);
        document.getElementById('out-energy').innerText = data.hash.energy.toFixed(6);
        document.getElementById('out-torsion').innerText = data.hash.torsion.toFixed(6);

        // Populate timeline
        const container = document.getElementById('history-container');
        container.innerHTML = '';
        
        data.history.forEach(step => {
            const div = document.createElement('div');
            div.className = 'timeline-step';
            div.innerHTML = `
                <p><strong>Iteración ${step.iteration}</strong></p>
                <p>Input <span class="math">n = ${step.input_n}</span> (Z7 = ${step.n_z7})</p>
                <p>Covarianza de entrelazamiento: <span class="math">${step.covariance.toFixed(6)}</span></p>
                <p>Δn Termodinámico: ${step.delta_n} → Nuevo <span class="math">n = ${step.next_n}</span></p>
            `;
            container.appendChild(div);
        });

        // Show results
        loader.classList.add('hidden');
        resultsPanel.classList.remove('hidden');

    } catch (err) {
        alert('Fallo de conexión.');
        loader.classList.add('hidden');
    }
});
