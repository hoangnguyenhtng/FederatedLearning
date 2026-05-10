// ===== Tab Navigation =====
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ===== Training Loss Data (from actual training_history.json) =====
const lossData = [
    1.456, 1.399, 1.366, 1.345, 1.334, 1.327, 1.322, 1.319, 1.312, 1.306,
    1.303, 1.298, 1.293, 1.291, 1.291, 1.290, 1.288, 1.289, 1.287, 1.286,
    1.285, 1.284, 1.284, 1.283, 1.282, 1.282, 1.283, 1.280, 1.280, 1.278,
    1.281, 1.279, 1.277, 1.277, 1.274, 1.276, 1.278, 1.273, 1.275, 1.275,
    1.273, 1.273, 1.269, 1.272, 1.273, 1.272, 1.274, 1.272, 1.272, 1.271,
    1.272, 1.271, 1.271, 1.274, 1.271, 1.269, 1.269, 1.271, 1.271, 1.269,
    1.269, 1.272, 1.272, 1.273, 1.274, 1.272, 1.270, 1.270, 1.269, 1.270,
    1.270, 1.269, 1.267, 1.269, 1.268, 1.270, 1.269, 1.269, 1.266, 1.267,
    1.269, 1.265, 1.267, 1.264, 1.265, 1.265, 1.266, 1.268, 1.267, 1.269,
    1.269, 1.268, 1.269, 1.270, 1.269, 1.268, 1.268, 1.268, 1.269, 1.268
];

const clientSamples = [127, 100, 90, 97, 99, 96, 87, 128, 105, 105];

// ===== Charts (initialized when tab becomes visible) =====
let lossChart, clientChart, ratingChart, fusionChart;
let chartsInitialized = false;

function initCharts() {
    if (chartsInitialized) return;
    chartsInitialized = true;

    const chartDefaults = {
        color: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.08)',
    };
    Chart.defaults.color = chartDefaults.color;
    Chart.defaults.borderColor = chartDefaults.borderColor;

    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 100}, (_, i) => i + 1),
                datasets: [{
                    label: 'Distributed Loss',
                    data: lossData,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99,102,241,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { 
                        backgroundColor: 'rgba(17,24,39,0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99,102,241,0.3)',
                        borderWidth: 1,
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Round' }, grid: { display: false } },
                    y: { title: { display: true, text: 'Loss' }, min: 1.25, max: 1.47 }
                }
            }
        });
    }

    // Client Distribution Chart
    const clientCtx = document.getElementById('clientChart');
    if (clientCtx) {
        clientChart = new Chart(clientCtx, {
            type: 'bar',
            data: {
                labels: clientSamples.map((_, i) => `Client ${i}`),
                datasets: [{
                    label: 'Samples',
                    data: clientSamples,
                    backgroundColor: clientSamples.map((_, i) =>
                        `hsla(${160 + i * 20}, 70%, 50%, 0.6)`
                    ),
                    borderColor: clientSamples.map((_, i) =>
                        `hsla(${160 + i * 20}, 70%, 50%, 1)`
                    ),
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Số lượng samples' } }
                }
            }
        });
    }

    // Rating Distribution
    const ratingCtx = document.getElementById('ratingChart');
    if (ratingCtx) {
        ratingChart = new Chart(ratingCtx, {
            type: 'doughnut',
            data: {
                labels: ['⭐ 1', '⭐ 2', '⭐ 3', '⭐ 4', '⭐ 5'],
                datasets: [{
                    data: [85, 52, 98, 312, 487],
                    backgroundColor: [
                        'rgba(239,68,68,0.7)', 'rgba(245,158,11,0.7)',
                        'rgba(234,179,8,0.7)', 'rgba(34,197,94,0.7)',
                        'rgba(16,185,129,0.7)'
                    ],
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'right', labels: { padding: 12, font: { size: 12 } } }
                }
            }
        });
    }
}

// Init fusion chart separately (different tab)
function initFusionChart() {
    const ctx = document.getElementById('fusionChart');
    if (!ctx || fusionChart) return;
    
    const tw = parseFloat(document.getElementById('textWeight').value);
    const iw = parseFloat(document.getElementById('imageWeight').value);
    const bw = parseFloat(document.getElementById('behaviorWeight').value);
    
    fusionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Text', 'Image', 'Behavior'],
            datasets: [{
                data: [tw, iw, bw],
                backgroundColor: ['rgba(99,102,241,0.8)', 'rgba(245,158,11,0.8)', 'rgba(16,185,129,0.8)'],
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            cutout: '55%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 16, font: { size: 13 } } }
            }
        }
    });
}

// Lazy-init charts when tabs are shown
const observer = new MutationObserver(() => {
    if (document.getElementById('tab-training').classList.contains('active')) initCharts();
    if (document.getElementById('tab-multimodal').classList.contains('active')) initFusionChart();
});
document.querySelectorAll('.tab-content').forEach(t => observer.observe(t, { attributes: true, attributeFilter: ['class'] }));

// ===== Fusion Weight Sliders =====
['textWeight', 'imageWeight', 'behaviorWeight'].forEach(id => {
    const slider = document.getElementById(id);
    if (!slider) return;
    slider.addEventListener('input', () => {
        const t = parseFloat(document.getElementById('textWeight').value);
        const i = parseFloat(document.getElementById('imageWeight').value);
        const b = parseFloat(document.getElementById('behaviorWeight').value);
        const total = t + i + b || 1;
        
        document.getElementById('textWeightVal').textContent = (t / total).toFixed(2);
        document.getElementById('imageWeightVal').textContent = (i / total).toFixed(2);
        document.getElementById('behaviorWeightVal').textContent = (b / total).toFixed(2);
        
        if (fusionChart) {
            fusionChart.data.datasets[0].data = [t, i, b];
            fusionChart.update();
        }
    });
});

// ===== DP Epsilon Slider =====
const dpSlider = document.getElementById('dpEpsilon');
if (dpSlider) {
    dpSlider.addEventListener('input', () => {
        const eps = parseFloat(dpSlider.value);
        document.getElementById('dpEpsilonVal').textContent = eps.toFixed(1);
        
        const privacy = Math.max(10, 100 - eps * 9);
        const utility = Math.min(95, 50 + eps * 5);
        
        document.getElementById('privacyBar').style.width = privacy + '%';
        document.getElementById('privacyBar').textContent = `Privacy: ${Math.round(privacy)}%`;
        document.getElementById('utilityBar').style.width = utility + '%';
        document.getElementById('utilityBar').textContent = `Utility: ${Math.round(utility)}%`;
        
        const explain = eps <= 1 
            ? `ε = ${eps.toFixed(1)} → Bảo mật RẤT CAO. Nhiễu lớn, model kém hơn nhưng data gần như không thể suy ra.`
            : eps <= 5 
            ? `ε = ${eps.toFixed(1)} → Cân bằng tốt giữa privacy và utility. Phù hợp production.`
            : `ε = ${eps.toFixed(1)} → Bảo mật THẤP. Ít nhiễu, accuracy cao nhưng rủi ro privacy lớn hơn.`;
        document.getElementById('dpExplain').textContent = explain;
    });
}

// ===== FL Animation Demo =====
const btnDemo = document.getElementById('btnPrivacyDemo');
if (btnDemo) {
    btnDemo.addEventListener('click', () => {
        const anim = document.getElementById('flAnimation');
        anim.classList.add('active');
        btnDemo.disabled = true;
        btnDemo.textContent = '⏳ Đang chạy...';
        runFLAnimation();
    });
}

async function runFLAnimation() {
    const clients = [
        document.getElementById('client0'),
        document.getElementById('client1'),
        document.getElementById('client2'),
    ];
    const statuses = [
        document.getElementById('client0Status'),
        document.getElementById('client1Status'),
        document.getElementById('client2Status'),
    ];
    const serverInfo = document.getElementById('serverInfo');
    
    for (let round = 1; round <= 3; round++) {
        serverInfo.textContent = `Round ${round}/3: Gửi global model...`;
        serverInfo.style.color = '#6366f1';
        await sleep(800);

        // Step 1: Clients receive model
        for (let i = 0; i < 3; i++) {
            statuses[i].textContent = '📥 Nhận model';
            statuses[i].style.background = 'rgba(99,102,241,0.2)';
            statuses[i].style.color = '#818cf8';
        }
        await sleep(600);

        // Step 2: Local training
        serverInfo.textContent = `Round ${round}/3: Clients đang train LOCAL...`;
        for (let i = 0; i < 3; i++) {
            clients[i].classList.add('training');
            clients[i].classList.remove('sending', 'done');
            statuses[i].textContent = '🔄 Training local...';
            statuses[i].style.background = 'rgba(245,158,11,0.2)';
            statuses[i].style.color = '#fbbf24';
        }
        await sleep(1500);

        // Step 3: Send only weights (NOT data!)
        for (let i = 0; i < 3; i++) {
            clients[i].classList.remove('training');
            clients[i].classList.add('sending');
            statuses[i].textContent = '📤 Gửi WEIGHTS (không data!)';
            statuses[i].style.background = 'rgba(99,102,241,0.2)';
            statuses[i].style.color = '#818cf8';
        }
        serverInfo.textContent = `Round ${round}/3: Nhận weights từ clients...`;
        await sleep(1000);

        // Step 4: Aggregate
        serverInfo.textContent = `Round ${round}/3: FedAvg aggregation...`;
        serverInfo.style.color = '#34d399';
        await sleep(800);

        // Step 5: Done
        for (let i = 0; i < 3; i++) {
            clients[i].classList.remove('sending');
            clients[i].classList.add('done');
            statuses[i].textContent = '✅ Done';
            statuses[i].style.background = 'rgba(16,185,129,0.2)';
            statuses[i].style.color = '#34d399';
        }
        serverInfo.textContent = `Round ${round}/3 hoàn thành! Loss giảm.`;
        await sleep(600);
    }

    serverInfo.textContent = '✅ Training hoàn tất! Data KHÔNG BAO GIỜ rời khỏi client.';
    serverInfo.style.color = '#34d399';
    
    const btn = document.getElementById('btnPrivacyDemo');
    btn.disabled = false;
    btn.textContent = '🔄 Chạy lại';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
