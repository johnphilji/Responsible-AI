document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    const resultsSection = document.getElementById('resultsSection');
    const errorMsg = document.getElementById('error-message');

    // UI Loading State
    btnText.textContent = "Processing...";
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    errorMsg.classList.add('hidden');
    resultsSection.classList.add('hidden');

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Process Response
            updateDecisionUI('normalDecisionUI', 'normalDecisionText', data.normal_ai_decision);
            updateDecisionUI('respDecisionUI', 'respDecisionText', data.responsible_ai_decision);

            // Confidence Bar
            const confBar = document.getElementById('confidenceBar');
            const confText = document.getElementById('confidenceText');
            confBar.style.width = `${data.confidence}%`;
            confText.textContent = `${data.confidence}%`;
            
            if(data.confidence >= 80) {
                confBar.style.backgroundColor = 'var(--approved)';
            } else if(data.confidence >= 50) {
                confBar.style.backgroundColor = 'var(--moderate)';
            } else {
                confBar.style.backgroundColor = 'var(--rejected)';
            }

            // Populate Explanations
            const explList = document.getElementById('explanationList');
            explList.innerHTML = ''; // clear

            if(data.explanations && data.explanations.length > 0) {
                data.explanations.forEach(expl => {
                    const row = document.createElement('div');
                    row.className = 'expl-row';
                    
                    const title = document.createElement('div');
                    title.className = 'expl-title';
                    title.textContent = `- ${expl.feature} (${expl.value}):`;
                    
                    const reason = document.createElement('div');
                    reason.className = `expl-reason ${expl.color}`;
                    reason.textContent = `${expl.text} → ${expl.impact} impact`;

                    row.appendChild(title);
                    row.appendChild(reason);
                    explList.appendChild(row);
                });
            }

            // Bias Check
            const biasContent = document.querySelector('.bias-content');
            if (data.bias.detected) {
                biasContent.innerHTML = `
                    <div class="bias-warning">⚠️ Potential bias detected in decision-making</div>
                    <p>Case A (${data.bias.original_case.split(':')[0]}): <strong>${data.bias.original_case.split(':')[1]}</strong></p>
                    <p>Case B (${data.bias.flipped_case.split(':')[0]}): <strong>${data.bias.flipped_case.split(':')[1]}</strong></p>
                    <p class="fair-note">The model changed its prediction based solely on Gender. This violates fairness thresholds and requires algorithmic adjustment.</p>
                `;
            } else {
                biasContent.innerHTML = `
                    <div class="bias-safe">✅ Bias test passed</div>
                    <p>Case A (${data.bias.original_case.split(':')[0]}): <strong>${data.bias.original_case.split(':')[1]}</strong></p>
                    <p>Case B (${data.bias.flipped_case.split(':')[0]}): <strong>${data.bias.flipped_case.split(':')[1]}</strong></p>
                    <p class="fair-note">Responsible AI ensures fairness checks across groups. The decision remained consistent regardless of demographic variations.</p>
                `;
            }

            // Show Results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            resultsSection.classList.remove('hidden');
        } else {
            throw new Error(data.detail || "Server error occurred");
        }

    } catch (err) {
        errorMsg.textContent = "Error: " + err.message;
        errorMsg.classList.remove('hidden');
    } finally {
        btnText.textContent = "Generate AI Predictions";
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});

function updateDecisionUI(uiId, textId, decisionStr) {
    const uiBox = document.getElementById(uiId);
    const textBox = document.getElementById(textId);
    
    // Clean up
    uiBox.classList.remove('approved', 'rejected');
    
    // Determine sentiment
    const strLower = decisionStr.toLowerCase();
    
    if (strLower.includes('yes') || strLower.includes('approved')) {
        textBox.textContent = strLower.includes('approved') ? "Approved" : "Yes";
        uiBox.classList.add('approved');
    } else if (strLower.includes('no') || strLower.includes('rejected')) {
        textBox.textContent = strLower.includes('rejected') ? "Rejected" : "No";
        uiBox.classList.add('rejected');
    } else {
        textBox.textContent = decisionStr; // Probably an error message from normal AI
    }
}
