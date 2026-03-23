document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    
    const resultsSection = document.getElementById('resultsSection');
    const errorMsg = document.getElementById('error-message');

    btnText.textContent = "Processing Risk...";
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
            
            // Extracted Decisions
            updateDecisionUI('normalDecisionUI', 'normalDecisionText', data.normal_ai_decision, 'normalDecisionIcon');
            updateDecisionUI('respDecisionUI', 'respDecisionText', data.responsible_ai_decision, 'respDecisionIcon');

            // Confidence Score UI
            const confBar = document.getElementById('confidenceBar');
            const confText = document.getElementById('confidenceText');
            confBar.style.width = `${data.confidence}%`;
            confText.textContent = `${data.confidence}%`;
            
            if(data.confidence >= 70) confBar.style.backgroundColor = 'var(--approved)';
            else if(data.confidence >= 40) confBar.style.backgroundColor = 'var(--review)';
            else confBar.style.backgroundColor = 'var(--rejected)';

            // Explanations Formatting
            const explList = document.getElementById('explanationList');
            explList.innerHTML = ''; 

            if(data.explanations && data.explanations.length > 0) {
                data.explanations.forEach(expl => {
                    const row = document.createElement('div');
                    row.className = 'expl-row';
                    
                    const title = document.createElement('div');
                    title.className = 'expl-title';
                    title.innerHTML = `- <strong>${expl.feature}</strong>:`;
                    
                    const reason = document.createElement('div');
                    reason.className = `expl-reason ${expl.color}`;
                    reason.innerHTML = `${expl.text} &rarr; <strong>${expl.impact}</strong>`;

                    row.appendChild(title);
                    row.appendChild(reason);
                    explList.appendChild(row);
                });
            }

            // Fairness Testing Setup
            const biasContent = document.querySelector('.bias-content');
            if (data.bias.detected) {
                let htmlBlock = `<div class="bias-warning">⚠️ Potential bias detected</div><ul>`;
                data.bias.messages.forEach(msg => {
                    htmlBlock += `<li>${msg.test}: <br>Base outcome: <strong>${msg.base}</strong> vs Flipped outcome: <strong>${msg.flipped}</strong></li>`;
                });
                htmlBlock += `</ul><p class="fair-note">Responsible AI requires fairness validation across demographic groups.</p>`;
                biasContent.innerHTML = htmlBlock;
            } else {
                biasContent.innerHTML = `
                    <div class="bias-safe">✅ Fairness simulation passed</div>
                    <p>Internal validations confirm swapping Gender or Property Area demographics continuously matched the Baseline ML prediction (<strong>${data.bias.base_case}</strong>).</p>
                    <p class="fair-note">Responsible AI requires fairness validation across demographic groups.</p>
                `;
            }

            // Unhide UI
            resultsSection.classList.remove('hidden');
            
            // Quick scroll for Mobile UX
            if(window.innerWidth < 1000) {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }

        } else {
            throw new Error("Server connectivity failed.");
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

function updateDecisionUI(uiId, textId, decisionStr, iconId) {
    const uiBox = document.getElementById(uiId);
    const textBox = document.getElementById(textId);
    const iconBox = document.getElementById(iconId);
    
    uiBox.classList.remove('approved', 'rejected', 'review');
    
    const strLower = typeof decisionStr === 'string' ? decisionStr.toLowerCase() : '';
    
    if (strLower.includes('yes') || strLower.includes('approved')) {
        textBox.textContent = strLower.includes('approved') ? "Approved" : "Approved";
        uiBox.classList.add('approved');
        if(iconBox) iconBox.textContent = '✓';
    } else if (strLower.includes('review') || strLower.includes('indeterminate')) {
        textBox.textContent = "Review";
        uiBox.classList.add('review');
        if(iconBox) iconBox.textContent = '⚠️';
    } else if (strLower.includes('no') || strLower.includes('rejected')) {
        textBox.textContent = strLower.includes('rejected') ? "Declined" : "Declined";
        uiBox.classList.add('rejected');
        if(iconBox) iconBox.textContent = '✕';
    } else {
        textBox.textContent = decisionStr; 
        if(iconBox) iconBox.textContent = '•';
    }
}
