/**
 * Vehicle Speed Tracker - Frontend Application Logic
 * Manages Webcam stream, canvas calibration clicking, WebSocket streaming,
 * and AI annotated frame rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("displayCanvas");
    const ctx = canvas.getContext("2d");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const calibrateBtn = document.getElementById("calibrateBtn");
    const calibrateBtnText = document.getElementById("calibrateBtnText");
    const clearBtn = document.getElementById("clearBtn");
    const instructionBanner = document.getElementById("instructionBanner");
    const metricCount = document.getElementById("metricCount");
    const metricMaxSpeed = document.getElementById("metricMaxSpeed");
    const metricFps = document.getElementById("metricFps");
    const vehicleList = document.getElementById("vehicleList");

    // State Variables
    let ws = null;
    let isCalibrating = false;
    let calibrationPoints = []; // Stores up to 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    let isSendingFrame = false;
    let frameCount = 0;
    let lastFpsCheck = performance.now();
    let currentFps = 0;

    // Mobile Device Detection & Rate Limiting
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;
    let lastSendTime = 0;

    // Offscreen canvas for frame capture
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = 1280;
    captureCanvas.height = 720;
    const captureCtx = captureCanvas.getContext("2d");

    /**
     * Initialize Webcam Feed
     */
    async function initWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: isMobile ? 720 : 1280 },
                    height: { ideal: isMobile ? 1280 : 720 },
                    facingMode: "environment"
                },
                audio: false
            });
            video.srcObject = stream;
            await video.play();

            // Set canvas size to native camera resolution
            canvas.width = video.videoWidth || (isMobile ? 720 : 1280);
            canvas.height = video.videoHeight || (isMobile ? 1280 : 720);

            // Scale capture canvas for mobile 4G bandwidth efficiency
            if (isMobile) {
                const maxDim = Math.max(canvas.width, canvas.height);
                const scale = 640 / maxDim;
                captureCanvas.width = Math.round(canvas.width * scale);
                captureCanvas.height = Math.round(canvas.height * scale);
            } else {
                captureCanvas.width = canvas.width;
                captureCanvas.height = canvas.height;
            }

            console.log(`Webcam initialized at ${canvas.width}x${canvas.height} (Capture: ${captureCanvas.width}x${captureCanvas.height})`);
            initWebSocket();
        } catch (err) {
            console.error("Camera access error:", err);
            statusText.textContent = "Camera Error: " + err.message;
            statusDot.classList.remove("connected");
            alert("Could not access webcam. Please ensure camera permissions are granted in your browser.");
        }
    }

    /**
     * Initialize WebSocket Connection to Backend
     */
    function initWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        statusText.textContent = "Connecting to Backend...";
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected.");
            statusDot.classList.add("connected");
            statusText.textContent = "AI Server Connected";
            startStreaming();
        };

        ws.onmessage = (event) => {
            isSendingFrame = false;
            try {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.warn("Backend warning:", data.error);
                    return;
                }

                if (data.frame) {
                    renderFrame(data.frame);
                }

                if (data.active_vehicles !== undefined || data.top_records !== undefined) {
                    updateMetricsAndList(data);
                }
            } catch (err) {
                console.error("Error processing frame:", err);
            }
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            statusDot.classList.remove("connected");
            statusText.textContent = "Connection Error";
        };

        ws.onclose = () => {
            console.warn("WebSocket connection closed. Reconnecting in 2 seconds...");
            statusDot.classList.remove("connected");
            statusText.textContent = "Disconnected (Retrying...)";
            setTimeout(initWebSocket, 2000);
        };
    }

    /**
     * Renders incoming base64 frame from server onto display canvas
     */
    function renderFrame(base64Image) {
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw calibration points overlay if user is actively calibrating
            if (isCalibrating || calibrationPoints.length > 0) {
                drawCalibrationOverlay();
            }
        };
        img.src = base64Image;
    }

    /**
     * Draws calibration points and polygon on the live canvas
     */
    function drawCalibrationOverlay() {
        if (calibrationPoints.length === 0) return;

        ctx.save();

        if (calibrationPoints.length > 1) {
            ctx.beginPath();
            ctx.moveTo(calibrationPoints[0][0], calibrationPoints[0][1]);
            for (let i = 1; i < calibrationPoints.length; i++) {
                ctx.lineTo(calibrationPoints[i][0], calibrationPoints[i][1]);
            }
            if (calibrationPoints.length === 4) {
                ctx.closePath();
                ctx.fillStyle = "rgba(0, 255, 128, 0.15)";
                ctx.fill();
            }
            ctx.strokeStyle = "#00ff80";
            ctx.lineWidth = 2.5;
            ctx.setLineDash([6, 4]);
            ctx.stroke();
        }

        calibrationPoints.forEach((pt, idx) => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 7, 0, 2 * Math.PI);
            ctx.fillStyle = "#00ff80";
            ctx.shadowColor = "#00ff80";
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();

            ctx.shadowBlur = 0;
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 14px 'Outfit', sans-serif";
            ctx.fillText(`P${idx + 1}`, pt[0] + 10, pt[1] - 8);
        });

        ctx.restore();
    }

    const speedLimitInput = document.getElementById("speedLimitInput");
    const roiDistanceInput = document.getElementById("roiDistanceInput");
    const violationLabel = document.getElementById("violationLabel");

    if (speedLimitInput && violationLabel) {
        speedLimitInput.addEventListener("input", () => {
            const val = parseFloat(speedLimitInput.value) || 50;
            violationLabel.textContent = `Violations (>${val}km/h)`;
        });
    }

    /**
     * Continuous Frame Streaming Loop via WebSocket with 4G Bandwidth Rate Limiting
     */
    function startStreaming() {
        function sendFrameLoop() {
            const now = Date.now();
            const minInterval = isMobile ? 65 : 30; // Max ~15 FPS on 4G mobile, 30 FPS on desktop

            if ((now - lastSendTime >= minInterval) && ws && ws.readyState === WebSocket.OPEN && !isSendingFrame) {
                if (video.readyState >= video.HAVE_CURRENT_DATA) {
                    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
                    const jpegQuality = isMobile ? 0.55 : 0.75;
                    const base64Data = captureCanvas.toDataURL("image/jpeg", jpegQuality);

                    const engineVal = document.getElementById("engineSelect") ? document.getElementById("engineSelect").value : "motion";
                    const speedLimitVal = speedLimitInput ? (parseFloat(speedLimitInput.value) || 50.0) : 50.0;
                    const roiDistanceVal = roiDistanceInput ? (parseFloat(roiDistanceInput.value) || 10.0) : 10.0;

                    const payload = {
                        image: base64Data,
                        points: calibrationPoints,
                        engine: engineVal,
                        speed_limit: speedLimitVal,
                        roi_distance: roiDistanceVal
                    };

                    lastSendTime = now;
                    isSendingFrame = true;
                    ws.send(JSON.stringify(payload));
                }
            }
            requestAnimationFrame(sendFrameLoop);
        }
        sendFrameLoop();
    }

    /**
     * Calibration Canvas Pointer & Mobile Touch Handler
     */
    function handleCalibrationInput(clientX, clientY) {
        if (!isCalibrating) return;

        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        const clickX = ((clientX - rect.left) / rect.width) * canvas.width;
        const clickY = ((clientY - rect.top) / rect.height) * canvas.height;

        const px = Math.max(0, Math.min(canvas.width, Math.round(clickX)));
        const py = Math.max(0, Math.min(canvas.height, Math.round(clickY)));

        if (calibrationPoints.length < 4) {
            calibrationPoints.push([px, py]);
            updateCalibrationUI();
        }

        if (calibrationPoints.length === 4) {
            isCalibrating = false;
            canvas.classList.remove("calibrating");
            calibrateBtn.classList.remove("active");
            calibrateBtnText.textContent = "Recalibrate";
            instructionBanner.style.display = "none";
        }
    }

    canvas.addEventListener("pointerdown", (event) => {
        handleCalibrationInput(event.clientX, event.clientY);
    });

    /**
     * Toggle Calibration Mode Button
     */
    calibrateBtn.addEventListener("click", () => {
        isCalibrating = !isCalibrating;

        if (isCalibrating) {
            if (calibrationPoints.length === 4) {
                calibrationPoints = [];
            }
            canvas.classList.add("calibrating");
            calibrateBtn.classList.add("active");
            calibrateBtnText.textContent = "Cancel Calibration";
            instructionBanner.style.display = "block";
            updateCalibrationUI();
        } else {
            canvas.classList.remove("calibrating");
            calibrateBtn.classList.remove("active");
            calibrateBtnText.textContent = calibrationPoints.length === 4 ? "Recalibrate" : "Calibrate";
            instructionBanner.style.display = "none";
        }
    });

    /**
     * Reset Calibration Button
     */
    clearBtn.addEventListener("click", () => {
        calibrationPoints = [];
        isCalibrating = false;
        canvas.classList.remove("calibrating");
        calibrateBtn.classList.remove("active");
        calibrateBtnText.textContent = "Calibrate";
        instructionBanner.style.display = "none";
        updateCalibrationUI();
    });

    /**
     * Update Calibration Instruction Text
     */
    function updateCalibrationUI() {
        const count = calibrationPoints.length;
        instructionBanner.textContent = `Click 4 corners on road to calibrate (${count}/4 points set)`;
    }

    /**
     * Update Dashboard Telemetry & Top 10 Vehicle Speed Log
     */
    function updateMetricsAndList(data) {
        const active = data.active_vehicles || [];
        const topRecords = data.top_records || [];
        const totalCount = data.total_count || 0;
        const speedingCount = data.speeding_count || 0;

        const metricCount = document.getElementById("metricCount");
        const metricMaxSpeed = document.getElementById("metricMaxSpeed");
        const metricTotal = document.getElementById("metricTotal");
        const metricViolations = document.getElementById("metricViolations");

        if (metricCount) metricCount.textContent = active.length;
        if (metricTotal) metricTotal.textContent = totalCount;
        if (metricViolations) metricViolations.textContent = speedingCount;

        let maxSpeed = 0;
        if (topRecords.length > 0) {
            maxSpeed = Math.max(...topRecords.map(r => r.max_speed));
            if (metricMaxSpeed) metricMaxSpeed.innerHTML = `${maxSpeed.toFixed(1)} <small style="font-size: 0.7rem;">km/h</small>`;
        } else if (metricMaxSpeed) {
            metricMaxSpeed.innerHTML = `0.0 <small style="font-size: 0.7rem;">km/h</small>`;
        }

        if (topRecords.length === 0) {
            vehicleList.innerHTML = `<div class="empty-state">No vehicles logged yet</div>`;
            return;
        }

        let html = "";
        topRecords.forEach(r => {
            const isSpeeding = r.is_speeding || r.max_speed > 50;
            html += `
                <div class="vehicle-item ${isSpeeding ? 'speeding' : ''}">
                    <div class="vehicle-info">
                        <span class="vehicle-name">${r.class} <small style="color: var(--accent-cyan);">#${r.id}</small></span>
                        <span class="vehicle-id">Last Speed: ${r.last_speed.toFixed(1)} km/h</span>
                    </div>
                    <div class="vehicle-speed">${r.max_speed.toFixed(1)} <small style="font-size: 0.7rem;">km/h max</small></div>
                </div>
            `;
        });
        vehicleList.innerHTML = html;
    }

    // Start application
    initWebcam();
});
