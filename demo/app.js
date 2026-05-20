// ===== Tab Navigation =====
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ===== Run config (đồng bộ với thí nghiệm: số client federated) =====
/** Số client tham gia FL — chỉnh 1 chỗ này khi đổi config train. */
const NUM_FEDERATED_CLIENTS = 40;

function mulberry32(seed) {
    return function () {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Số sample / client (minh họa phân bố non-IID, deterministic theo seed). */
function buildClientSamples(numClients, seed = 20260510) {
    const rnd = mulberry32(seed);
    const out = [];
    for (let i = 0; i < numClients; i++) {
        out.push(60 + Math.floor(rnd() * 95));
    }
    return out;
}

const clientSamples = buildClientSamples(NUM_FEDERATED_CLIENTS);

function syncDemoNumbersFromConfig() {
    const sc = document.getElementById('statClients');
    if (sc) sc.textContent = String(NUM_FEDERATED_CLIENTS);
    const total = clientSamples.reduce((a, b) => a + b, 0);
    const ss = document.getElementById('statSamples');
    if (ss) ss.textContent = total.toLocaleString('en-US');
    const cap = document.getElementById('captionNumClients');
    if (cap) cap.textContent = String(NUM_FEDERATED_CLIENTS);
    const hint = document.getElementById('clientChartHintN');
    if (hint) hint.textContent = String(NUM_FEDERATED_CLIENTS);
    for (let i = 0; i < 3; i++) {
        const card = document.getElementById(`client${i}`);
        if (!card) continue;
        const dataEl = card.querySelector('.client-data');
        if (dataEl) dataEl.textContent = `${clientSamples[i]} samples`;
    }
}
syncDemoNumbersFromConfig();

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
                        `hsla(${(160 + i * 5) % 360}, 70%, 50%, 0.6)`
                    ),
                    borderColor: clientSamples.map((_, i) =>
                        `hsla(${(160 + i * 5) % 360}, 70%, 50%, 1)`
                    ),
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 24,
                            font: { size: 9 },
                        },
                    },
                    y: { beginAtZero: true, title: { display: true, text: 'Số lượng samples' } },
                },
            },
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

function nowHMS() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
}

function ensureFlLogVisible() {
    const wrap = document.getElementById('flLogWrap');
    if (wrap) wrap.style.display = 'block';
}

function appendFlLog(line) {
    const el = document.getElementById('flLog');
    if (!el) return;
    ensureFlLogVisible();
    const txt = `[${nowHMS()}] ${line}\n`;
    el.textContent += txt;
    el.scrollTop = el.scrollHeight;
}

function clearFlLog() {
    const el = document.getElementById('flLog');
    if (el) el.textContent = '';
}

const btnClear = document.getElementById('btnFlLogClear');
if (btnClear) btnClear.addEventListener('click', clearFlLog);

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

    clearFlLog();
    appendFlLog(`FL server: start_simulation(num_clients=${NUM_FEDERATED_CLIENTS}, strategy=FedAvg, dp_sgd=on)`);
    appendFlLog(`Transport: encrypted channel (TLS) — payload=Δw (no raw data)`);
    appendFlLog(`DP-SGD: clip_norm=1.0, ε≈${(parseFloat(document.getElementById('dpEpsilon')?.value || '1.0') || 1.0).toFixed(1)} (demo slider)`);
    
    for (let round = 1; round <= 3; round++) {
        serverInfo.textContent = `Round ${round}/3: Gửi global model tới ${NUM_FEDERATED_CLIENTS} clients (minh họa 3)...`;
        serverInfo.style.color = '#6366f1';
        appendFlLog(`Round ${round}: broadcast global_model → ${NUM_FEDERATED_CLIENTS} clients`);
        await sleep(800);

        // Step 1: Clients receive model
        for (let i = 0; i < 3; i++) {
            statuses[i].textContent = '📥 Nhận model';
            statuses[i].style.background = 'rgba(99,102,241,0.2)';
            statuses[i].style.color = '#818cf8';
        }
        appendFlLog(`Round ${round}: clients[0..2] recv global_model (demo)`);
        await sleep(600);

        // Step 2: Local training
        serverInfo.textContent = `Round ${round}/3: ${NUM_FEDERATED_CLIENTS} clients train LOCAL (minh họa 3)...`;
        for (let i = 0; i < 3; i++) {
            clients[i].classList.add('training');
            clients[i].classList.remove('sending', 'done');
            statuses[i].textContent = '🔄 Training local...';
            statuses[i].style.background = 'rgba(245,158,11,0.2)';
            statuses[i].style.color = '#fbbf24';
        }
        appendFlLog(`Round ${round}: local_train(epochs=1) on-device — raw data stays local ✅`);
        await sleep(1500);

        // Step 3: Send only weights (NOT data!)
        for (let i = 0; i < 3; i++) {
            clients[i].classList.remove('training');
            clients[i].classList.add('sending');
            statuses[i].textContent = '📤 Gửi WEIGHTS (không data!)';
            statuses[i].style.background = 'rgba(99,102,241,0.2)';
            statuses[i].style.color = '#818cf8';
        }
        serverInfo.textContent = `Round ${round}/3: Nhận weights từ ${NUM_FEDERATED_CLIENTS} clients...`;
        appendFlLog(`Round ${round}: collect_fit_results(fraction_fit=0.6) — recv Δw from selected clients`);
        appendFlLog(`Round ${round}: apply DP noise + clip gradients (demo)`);
        await sleep(1000);

        // Step 4: Aggregate
        serverInfo.textContent = `Round ${round}/3: FedAvg aggregation...`;
        serverInfo.style.color = '#34d399';
        appendFlLog(`Round ${round}: aggregate_fit(strategy=FedAvg) → update global_model`);
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
        const fakeLoss = (1.45 - round * 0.06 + (round === 2 ? 0.01 : 0.0)).toFixed(3);
        appendFlLog(`Round ${round}: metrics — loss_distributed=${fakeLoss} (demo)`);
        await sleep(600);
    }

    serverInfo.textContent = '✅ Training hoàn tất! Data KHÔNG BAO GIỜ rời khỏi client.';
    serverInfo.style.color = '#34d399';
    appendFlLog(`DONE: training completed. Server only saw Δw + metadata; no raw reviews/images/purchases.`);
    
    const btn = document.getElementById('btnPrivacyDemo');
    btn.disabled = false;
    btn.textContent = '🔄 Chạy lại';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ===== Privacy Inspector: Noise Text Animation =====
function animateNoiseText() {
    const chars = '█▓▒░▄▀■□●◌◆◇▲△▼▽';
    const ids = ['noiseUser', 'noiseReview', 'noiseRating'];
    const lengths = [10, 18, 6];
    
    setInterval(() => {
        ids.forEach((id, i) => {
            const el = document.getElementById(id);
            if (!el) return;
            let s = '';
            for (let j = 0; j < lengths[i]; j++) {
                s += chars[Math.floor(Math.random() * chars.length)];
            }
            el.textContent = s;
        });
    }, 300);
}
animateNoiseText();

// ===== Privacy Inspector: Gradient Charts =====
let gradientChartsInit = false;
function initGradientCharts() {
    if (gradientChartsInit) return;
    const origCtx = document.getElementById('gradientOriginal');
    const noiseCtx = document.getElementById('gradientNoised');
    if (!origCtx || !noiseCtx) return;
    gradientChartsInit = true;

    // Generate deterministic "gradient" values
    const seed = 42;
    const n = 30;
    const origData = [];
    const noisedData = [];
    let s = seed;
    for (let i = 0; i < n; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        const v = ((s / 0x7fffffff) - 0.5) * 0.02; // small gradient values
        origData.push(v);
        // Add Gaussian-like noise (Box-Muller approximation)
        const u1 = Math.random(), u2 = Math.random();
        const noise = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2) * 0.015;
        noisedData.push(v + noise);
    }

    const labels = Array.from({length: n}, (_, i) => `w${i}`);
    const barDefaults = {
        responsive: true,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
            x: { display: false },
            y: { min: -0.04, max: 0.04, ticks: { font: { size: 9 }, color: '#64748b' },
                 grid: { color: 'rgba(255,255,255,0.04)' } }
        },
        animation: { duration: 800 }
    };

    new Chart(origCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: origData,
                backgroundColor: origData.map(v =>
                    v >= 0 ? 'rgba(99,102,241,0.7)' : 'rgba(245,158,11,0.7)'
                ),
                borderRadius: 2,
            }]
        },
        options: barDefaults
    });

    new Chart(noiseCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: noisedData,
                backgroundColor: noisedData.map(v =>
                    v >= 0 ? 'rgba(99,102,241,0.4)' : 'rgba(245,158,11,0.4)'
                ),
                borderRadius: 2,
            }]
        },
        options: barDefaults
    });
}

// Init gradient charts when privacy tab is shown
const privacyObserver = new MutationObserver(() => {
    const privacyTab = document.getElementById('tab-privacy');
    if (privacyTab && privacyTab.classList.contains('active')) {
        initGradientCharts();
    }
});
document.querySelectorAll('.tab-content').forEach(t =>
    privacyObserver.observe(t, { attributes: true, attributeFilter: ['class'] })
);

// ===== Privacy Inspector: Reconstruction Attack Demo =====
const btnAttack = document.getElementById('btnAttackDemo');
if (btnAttack) {
    btnAttack.addEventListener('click', () => {
        btnAttack.disabled = true;
        btnAttack.textContent = '⏳ Đang tấn công...';
        const resultDiv = document.getElementById('attackResult');
        const outputDiv = document.getElementById('attackOutput');
        const barDiv = document.getElementById('attackBar');
        resultDiv.style.display = 'block';
        outputDiv.innerHTML = '';
        barDiv.style.width = '0%';
        runAttackDemo(outputDiv, barDiv, btnAttack);
    });
}

async function runAttackDemo(outputDiv, barDiv, btn) {
    const steps = [
        { pct: 10, text: '🔍 Thu thập model weights từ server...', cls: '' },
        { pct: 25, text: '📐 Phân tích gradient structure (30 parameters)...', cls: '' },
        { pct: 40, text: '🧮 Thử gradient inversion attack (DLG method)...', cls: '' },
        { pct: 55, text: '❌ Gradient đã bị clip (max_norm=1.0) → không đủ thông tin', cls: 'fail' },
        { pct: 65, text: '🧮 Thử membership inference attack...', cls: '' },
        { pct: 75, text: '❌ DP noise (ε=1.0, σ=1.1) làm sai lệch kết quả → accuracy ~50% (random)', cls: 'fail' },
        { pct: 85, text: '🧮 Thử model inversion attack trên personal head...', cls: '' },
        { pct: 90, text: '❌ Personal head KHÔNG BAO GIỜ rời client → không có dữ liệu để tấn công', cls: 'fail' },
        { pct: 100, text: '🛡️ KẾT LUẬN: Tất cả tấn công thất bại! Dữ liệu người dùng được bảo vệ bởi 3 lớp: FedPer + DP-SGD + Gradient Clipping', cls: 'success' },
    ];

    for (const step of steps) {
        barDiv.style.width = step.pct + '%';
        const div = document.createElement('div');
        div.className = 'attack-step ' + step.cls;
        div.textContent = step.text;
        outputDiv.appendChild(div);
        await sleep(800);
    }

    btn.disabled = false;
    btn.textContent = '🔄 Thử Lại';
}
