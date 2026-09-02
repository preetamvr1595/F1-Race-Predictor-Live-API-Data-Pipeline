/* ==========================================================================
   F1 AI TELEMETRY ENGINE — INTERACTIVE FRONTEND APP LOGIC
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Initialize Chart.js Font Default
    if (window.Chart) {
        Chart.defaults.font.family = "'Outfit', sans-serif";
    }

    // Initialize Lucide Icons
    lucide.createIcons();

    // Global App State
    let driversList = [];
    let circuitsList = [];
    let gaugeChart = null;
    let radarChart = null;
    let comparisonChart = null;

    // DOM Elements
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    const driverSelect = document.getElementById("driver-select");
    const circuitSelect = document.getElementById("circuit-select");
    const gridSlider = document.getElementById("grid-slider");
    const gridValDisplay = document.getElementById("grid-val-display");
    const predictBtn = document.getElementById("predict-btn");

    const simCircuitSelect = document.getElementById("sim-circuit-select");
    const simulateBtn = document.getElementById("simulate-btn");

    const duelD1 = document.getElementById("duel-d1");
    const duelD2 = document.getElementById("duel-d2");

    // Initialize Application
    initApp();

    async function initApp() {
        await fetchDrivers();
        await fetchCircuits();
        setupEventListeners();
    }

    // Tab Switching Logic
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetTab = document.getElementById(`${tab.dataset.tab}-tab`);
            if (targetTab) targetTab.classList.add("active");
        });
    });

    // Fetch Drivers List
    async function fetchDrivers() {
        try {
            const res = await fetch("/api/drivers");
            driversList = await res.json();

            populateDriverSelects();
            updateDriverSummary();
        } catch (err) {
            console.error("Error fetching drivers:", err);
        }
    }

    // Fetch Circuits List
    async function fetchCircuits() {
        try {
            const res = await fetch("/api/circuits");
            circuitsList = await res.json();

            populateCircuitSelects();
        } catch (err) {
            console.error("Error fetching circuits:", err);
        }
    }

    // Populate Dropdown Selects
    function populateDriverSelects() {
        driverSelect.innerHTML = "";
        duelD1.innerHTML = "";
        duelD2.innerHTML = "";

        driversList.forEach((d, idx) => {
            const opt = document.createElement("option");
            opt.value = d.id;
            opt.textContent = `${d.flag} ${d.name} (${d.team})`;

            driverSelect.appendChild(opt.cloneNode(true));
            duelD1.appendChild(opt.cloneNode(true));

            const opt2 = opt.cloneNode(true);
            if (idx === 1) opt2.selected = true;
            duelD2.appendChild(opt2);
        });

        // Set default to Verstappen if available
        const defaultIndex = driversList.findIndex(d => d.id === "max_verstappen");
        if (defaultIndex !== -1) driverSelect.selectedIndex = defaultIndex;
    }

    function populateCircuitSelects() {
        circuitSelect.innerHTML = "";
        simCircuitSelect.innerHTML = "";

        circuitsList.forEach((c, idx) => {
            const opt = document.createElement("option");
            opt.value = c.id;
            opt.textContent = `${c.location} - ${c.name}`;

            circuitSelect.appendChild(opt.cloneNode(true));
            simCircuitSelect.appendChild(opt.cloneNode(true));
        });

        const monacoIdx = circuitsList.findIndex(c => c.id === "monaco");
        if (monacoIdx !== -1) {
            circuitSelect.selectedIndex = monacoIdx;
            simCircuitSelect.selectedIndex = monacoIdx;
        }
    }

    // Setup Event Listeners
    function setupEventListeners() {
        driverSelect.addEventListener("change", updateDriverSummary);
        gridSlider.addEventListener("input", (e) => {
            gridValDisplay.textContent = `P${e.target.value}${e.target.value == 1 ? ' (Pole)' : ''}`;
        });

        predictBtn.addEventListener("click", runPrediction);
        simulateBtn.addEventListener("click", runGridSimulation);

        duelD1.addEventListener("change", renderDriverDuel);
        duelD2.addEventListener("change", renderDriverDuel);

        // Initial render for duel
        renderDriverDuel();
    }

    // Update Driver Summary Info Box
    function updateDriverSummary() {
        const selectedId = driverSelect.value;
        const driver = driversList.find(d => d.id === selectedId);

        if (!driver) return;

        document.getElementById("summary-flag").textContent = driver.flag;
        document.getElementById("summary-name").textContent = driver.name;
        document.getElementById("summary-team").textContent = driver.team;
        document.getElementById("summary-points").textContent = `${driver.season_points.toFixed(1)} pts`;
        document.getElementById("summary-form").textContent = `P${driver.recent_form_position.toFixed(1)}`;
        document.getElementById("summary-dnf").textContent = `${(driver.dnf_rate * 100).toFixed(1)}%`;
        document.getElementById("summary-track").textContent = `P${driver.recent_form_position.toFixed(1)}`;
    }

    // Run Predictor API Call
    async function runPrediction() {
        const driverId = driverSelect.value;
        const circuitId = circuitSelect.value;
        const gridPos = parseInt(gridSlider.value);

        predictBtn.disabled = true;
        predictBtn.innerHTML = `<i data-lucide="loader" class="animate-spin"></i> RUNNING AI TELEMETRY...`;
        lucide.createIcons();

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    driver_id: driverId,
                    circuit_id: circuitId,
                    grid_position: gridPos
                })
            });

            const data = await res.json();
            renderPredictionResults(data);
        } catch (err) {
            console.error("Prediction API Error:", err);
            alert("Error running prediction. Please check inputs.");
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = `<i data-lucide="play-circle"></i> RUN AI PREDICTION ENGINE`;
            lucide.createIcons();
        }
    }

    // Render Prediction Visualizations
    function renderPredictionResults(data) {
        document.getElementById("results-section").classList.remove("hidden");
        document.getElementById("results-section").scrollIntoView({ behavior: 'smooth' });

        document.getElementById("kpi-podium").textContent = `${(data.podium_probability * 100).toFixed(1)}%`;
        document.getElementById("kpi-win").textContent = `${(data.winner_probability * 100).toFixed(1)}%`;
        document.getElementById("kpi-finish").textContent = `P${data.predicted_position}`;

        renderGaugeChart(data.podium_probability * 100, data.winner_probability * 100);
        renderRadarChart(data.radar_scores);
    }

    // Chart.js Gauge Chart
    function renderGaugeChart(podiumPct, winPct) {
        const ctx = document.getElementById("gaugeChart").getContext("2d");

        if (gaugeChart) gaugeChart.destroy();

        gaugeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Podium Probability (%)', 'Win Probability (%)'],
                datasets: [{
                    label: 'Probability',
                    data: [podiumPct.toFixed(1), winPct.toFixed(1)],
                    backgroundColor: ['#FFD700', '#00F0FF'],
                    borderRadius: 8,
                    barThickness: 32
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94A3B8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#F1F5F9', font: { weight: 'bold' } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => ` ${context.raw}%`
                        }
                    }
                }
            }
        });
    }

    // Chart.js Radar Chart
    function renderRadarChart(scores) {
        const ctx = document.getElementById("radarChart").getContext("2d");

        if (radarChart) radarChart.destroy();

        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Grid Position', 'Recent Form', 'Team Pace', 'Track History', 'Reliability'],
                datasets: [{
                    label: 'Telemetry Score',
                    data: [scores.grid, scores.form, scores.team_pace, scores.track_history, scores.reliability],
                    backgroundColor: 'rgba(255, 24, 1, 0.25)',
                    borderColor: '#FF1801',
                    borderWidth: 2,
                    pointBackgroundColor: '#FF1801',
                    pointBorderColor: '#FFF'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#F1F5F9', font: { size: 12, weight: 'bold' } },
                        ticks: { display: false },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // Run Full Grid Simulation API Call
    async function runGridSimulation() {
        const circuitId = simCircuitSelect.value;

        simulateBtn.disabled = true;
        simulateBtn.innerHTML = `<i data-lucide="loader" class="animate-spin"></i> SIMULATING GRAND PRIX...`;
        lucide.createIcons();

        try {
            const res = await fetch("/api/simulate-grid", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ circuit_id: circuitId })
            });

            const data = await res.json();
            renderSimulationResults(data.results);
        } catch (err) {
            console.error("Simulation API Error:", err);
            alert("Error running grid simulation.");
        } finally {
            simulateBtn.disabled = false;
            simulateBtn.innerHTML = `<i data-lucide="zap"></i> RUN FULL GRID SIMULATION`;
            lucide.createIcons();
        }
    }

    // Render Full Grid Simulation
    function renderSimulationResults(results) {
        document.getElementById("podium-showcase").classList.remove("hidden");
        document.getElementById("leaderboard-section").classList.remove("hidden");

        const r1 = results[0];
        const r2 = results[1];
        const r3 = results[2];

        document.getElementById("p1-name").textContent = `${r1.flag} ${r1.driver_name}`;
        document.getElementById("p1-team").textContent = r1.team;
        document.getElementById("p1-prob").textContent = `Win Chance: ${(r1.winner_probability * 100).toFixed(1)}%`;

        document.getElementById("p2-name").textContent = `${r2.flag} ${r2.driver_name}`;
        document.getElementById("p2-team").textContent = r2.team;
        document.getElementById("p2-prob").textContent = `Podium: ${(r2.podium_probability * 100).toFixed(1)}%`;

        document.getElementById("p3-name").textContent = `${r3.flag} ${r3.driver_name}`;
        document.getElementById("p3-team").textContent = r3.team;
        document.getElementById("p3-prob").textContent = `Podium: ${(r3.podium_probability * 100).toFixed(1)}%`;

        const body = document.getElementById("leaderboard-body");
        body.innerHTML = "";

        results.forEach((item) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>P${item.predicted_rank}</strong></td>
                <td>${item.flag} ${item.driver_name}</td>
                <td>${item.team}</td>
                <td>P${item.start_grid}</td>
                <td><strong class="text-gold">${(item.podium_probability * 100).toFixed(1)}%</strong></td>
                <td><span class="text-cyan">${(item.winner_probability * 100).toFixed(1)}%</span></td>
            `;
            body.appendChild(tr);
        });
    }

    // Render Driver Telemetry Duel Chart
    function renderDriverDuel() {
        const d1Id = duelD1.value;
        const d2Id = duelD2.value;

        const d1 = driversList.find(d => d.id === d1Id);
        const d2 = driversList.find(d => d.id === d2Id);

        if (!d1 || !d2) return;

        const ctx = document.getElementById("comparisonChart").getContext("2d");

        if (comparisonChart) comparisonChart.destroy();

        comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Season Points', 'Recent Form Pos (Lower is better)', 'Recent Form Pts', 'DNF Rate (%)'],
                datasets: [
                    {
                        label: `${d1.flag} ${d1.name}`,
                        data: [d1.season_points, d1.recent_form_position, d1.recent_form_points, (d1.dnf_rate * 100).toFixed(1)],
                        backgroundColor: '#FF1801',
                        borderRadius: 6
                    },
                    {
                        label: `${d2.flag} ${d2.name}`,
                        data: [d2.season_points, d2.recent_form_position, d2.recent_form_points, (d2.dnf_rate * 100).toFixed(1)],
                        backgroundColor: '#00F0FF',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } }
                },
                plugins: {
                    legend: { labels: { color: '#F1F5F9', font: { size: 14, weight: 'bold' } } }
                }
            }
        });
    }
});
