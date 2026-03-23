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

            // Populate Explanations
            const explList = document.getElementById('explanationList');
            explList.innerHTML = ''; // clear

            if(data.explanations && data.explanations.length > 0) {
                data.explanations.forEach(expl => {
                    const li = document.createElement('li');
                    
                    const textNode = document.createElement('span');
                    textNode.innerHTML = `Because your <strong>${expl.feature}</strong> is <strong>${expl.value}</strong>, it had a `;
                    
                    const badge = document.createElement('span');
                    badge.className = 'impact-badge';
                    badge.textContent = `${expl.impact} impact`;
                    
                    const endNode = document.createTextNode(' on this decision.');

                    li.appendChild(textNode);
                    li.appendChild(badge);
                    li.appendChild(endNode);
                    explList.appendChild(li);
                });
            } else {
                explList.innerHTML = '<li>We looked at all your details, but no single specific factor strongly stood out to cause this.</li>';
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
