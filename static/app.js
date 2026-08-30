/**
 * Vehicle Speed Tracker - Ultra-Smooth 60 FPS Frontend Application Logic
 * Renders local camera feed at 60 FPS in hardware with zero visual latency,
 * sends lightweight downscaled frames to backend, and renders vector HUD
 * (bounding boxes, speed badges, trails, and BEV mini-map radar) directly on canvas.
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
    const engineSelect = document.getElementById("engineSelect");
    const speedLimitInput = document.getElementById("speedLimitInput");
    const roiDistanceInput = document.getElementById("roiDistanceInput");
    const violationLabel = document.getElementById("violationLabel");
    const vehicleList = document.getElementById("vehicleList");

    // State Variables
    let ws = null;
    let isCalibrating = false;
    let calibrationPoints = []; // [[x, y], ...] in canvas pixel coordinates
    let isSendingFrame = false;
    let lastSendTime = 0;
    let latestTelemetry = {
        calibrated: false,
        active_vehicles: [],
        bev_points: [],
        top_records: [],
        total_count: 0,
        speeding_count: 0
    };

    // Mobile Device Detection & Capture Resolution
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;

    // Compact Offscreen canvas for fast, lightweight AI frame extraction
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = 416;
    captureCanvas.height = 416;
    const captureCtx = captureCanvas.getContext("2d");

    /**
     * Initialize Webcam Feed with Robust Mobile Fallbacks
     */
    async function initWebcam() {
        video.muted = true;
        video.playsInline = true;
        video.setAttribute("playsinline", "true");
        video.setAttribute("webkit-playsinline", "true");
        video.setAttribute("muted", "true");
        video.setAttribute("autoplay", "true");

        try {
            let stream;
            try {
                // Try back camera with ideal resolution first
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: { ideal: "environment" },
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                });
            } catch (err1) {
                console.warn("Retrying with simple environment constraint...", err1);
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: "environment" },
                        audio: false
                    });
                } catch (err2) {
                    console.warn("Retrying with standard video constraint...", err2);
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: false
                    });
                }
            }

            video.srcObject = stream;
            try {
                await video.play();
            } catch (playErr) {
                console.warn("Autoplay blocked, will resume on touch:", playErr);
            }

            // Wait for camera stream metadata to ensure accurate native dimensions
            if (!video.videoWidth || !video.videoHeight) {
                await new Promise((resolve) => {
                    const onReady = () => {
                        video.removeEventListener("loadedmetadata", onReady);
                        video.removeEventListener("canplay", onReady);
                        resolve();
                    };
                    video.addEventListener("loadedmetadata", onReady);
                    video.addEventListener("canplay", onReady);
                    setTimeout(resolve, 800);
                });
            }

            // Set display canvas size to match native camera stream resolution
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;

            // Scale compact capture canvas maintaining aspect ratio
            const aspect = canvas.width / canvas.height;
            if (aspect >= 1) {
                captureCanvas.width = 416;
                captureCanvas.height = Math.max(200, Math.round(416 / aspect));
            } else {
                captureCanvas.height = 416;
                captureCanvas.width = Math.max(200, Math.round(416 * aspect));
            }

            console.log(`Camera active: Display ${canvas.width}x${canvas.height}, AI Capture ${captureCanvas.width}x${captureCanvas.height}`);

            // Start 60 FPS Hardware Render Loop & WebSocket
            requestAnimationFrame(renderLoop);
            initWebSocket();
        } catch (err) {
            console.error("Camera access error:", err);
            statusText.textContent = "Camera Error: " + err.message;
            statusDot.classList.remove("connected");
            alert("Could not access camera. Please ensure camera permissions are allowed in your browser settings.");
        }
    }

    // Touch/click listener to unlock video playback on strict mobile browsers
    const unlockVideo = () => {
        if (video.srcObject && video.paused) {
            video.play().catch(() => {});
        }
    };
    window.addEventListener("touchstart", unlockVideo, { passive: true });
    window.addEventListener("click", unlockVideo, { passive: true });

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
            statusText.textContent = "AI Telemetry Stream Active";
            startStreaming();
        };

        ws.onmessage = (event) => {
            isSendingFrame = false;
            try {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.warn("Backend notice:", data.error);
                    return;
                }

                // Update latest AI telemetry data
                latestTelemetry = data;
                updateMetricsAndList(data);
            } catch (err) {
                console.error("Error parsing telemetry JSON:", err);
            }
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            statusDot.classList.remove("connected");
            statusText.textContent = "Connection Error";
        };

        ws.onclose = () => {
            console.warn("WebSocket closed. Reconnecting in 2s...");
            statusDot.classList.remove("connected");
            statusText.textContent = "Disconnected (Retrying...)";
            setTimeout(initWebSocket, 2000);
        };
    }

    /**
     * 🚀 60 FPS Hardware Video & Vector HUD Render Loop
     * Renders crisp vector HUD overlays (bounding boxes, speed pills, trails, radar)
     * floating on top of the native hardware camera video stream.
     */
    function renderLoop() {
        // Clear HUD overlay canvas (transparent background over native video element)
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const speedLimit = speedLimitInput ? (parseFloat(speedLimitInput.value) || 50.0) : 50.0;
        const roiDistance = roiDistanceInput ? (parseFloat(roiDistanceInput.value) || 10.0) : 10.0;

        // 2. Draw Calibration Polygon & Handles
        drawCalibrationHUD(roiDistance);

        // 3. Draw Vehicle Overlays (Bounding boxes, motion trails, speed badges)
        drawVehicleOverlays(speedLimit);

        // 4. Draw Bird's-Eye View (BEV) Mini-Map Radar on Top-Right
        drawBEVRadar(roiDistance, speedLimit);

        // Request next 60 FPS animation frame
        requestAnimationFrame(renderLoop);
    }

    /**
     * Draws calibration polygon, region of interest tint, and clickable corner points
     */
    function drawCalibrationHUD(roiDistance) {
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

                // Labeled ROI distance marker
                ctx.fillStyle = "#00ff80";
                ctx.font = "bold 13px 'JetBrains Mono', monospace";
                ctx.fillText(`ROI Calibrated (${roiDistance.toFixed(1)}m)`, calibrationPoints[0][0], Math.max(25, calibrationPoints[0][1] - 12));
            }

            ctx.strokeStyle = "#00ff80";
            ctx.lineWidth = 2.5;
            ctx.setLineDash(calibrationPoints.length === 4 ? [] : [6, 4]);
            ctx.stroke();
        }

        // Draw handle corner pins
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
            ctx.font = "bold 13px 'Outfit', sans-serif";
            ctx.fillText(`P${idx + 1}`, pt[0] + 10, pt[1] - 8);
        });

        ctx.restore();
    }

    /**
     * Draws vector HUD overlays for active vehicles (bounding boxes, speed pills, trails)
     */
    function drawVehicleOverlays(speedLimit) {
        const vehicles = latestTelemetry.active_vehicles || [];
        if (vehicles.length === 0) return;

        ctx.save();

        vehicles.forEach(v => {
            const bbox = v.bbox || [0, 0, 0, 0];
            const x1 = bbox[0] * canvas.width;
            const y1 = bbox[1] * canvas.height;
            const x2 = bbox[2] * canvas.width;
            const y2 = bbox[3] * canvas.height;
            const bw = Math.max(10, x2 - x1);
            const bh = Math.max(10, y2 - y1);
            const speed = v.speed || 0.0;

            // Determine color palette based on speed
            let themeColor = "#00e676"; // Emerald Green
            if (speed > speedLimit) {
                themeColor = "#ff3d00"; // Crimson Red
            } else if (speed > (speedLimit * 0.8)) {
                themeColor = "#ffb300"; // Amber Warning
            }

            // Draw Motion Trail
            if (v.trail && v.trail.length > 1) {
                ctx.beginPath();
                ctx.moveTo(v.trail[0][0] * canvas.width, v.trail[0][1] * canvas.height);
                for (let i = 1; i < v.trail.length; i++) {
                    ctx.lineTo(v.trail[i][0] * canvas.width, v.trail[i][1] * canvas.height);
                }
                ctx.strokeStyle = themeColor;
                ctx.lineWidth = 2.5;
                ctx.shadowColor = themeColor;
                ctx.shadowBlur = 6;
                ctx.stroke();
                ctx.shadowBlur = 0;
            }

            // Draw Bounding Box with Corner Accents
            ctx.strokeStyle = themeColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, bw, bh);

            // Center Contact Point Dot
            const cx = (x1 + x2) / 2;
            const cy = y2;
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "#ff00e5";
            ctx.fill();

            // Draw High-Contrast Speed & Class Badge Pill
            const speedText = (speed > 0 || (v.trail && v.trail.length >= 2)) ? `${speed.toFixed(1)} km/h` : "Tracking...";
            const badgeLabel = `#${v.id} ${v.class} | ${speedText}`;
            ctx.font = "bold 12px 'JetBrains Mono', monospace";
            const textMetrics = ctx.measureText(badgeLabel);
            const badgeW = textMetrics.width + 16;
            const badgeH = 22;
            const badgeX = x1;
            const badgeY = Math.max(0, y1 - badgeH - 4);

            // Badge Background
            ctx.fillStyle = themeColor;
            ctx.fillRect(badgeX, badgeY, badgeW, badgeH);

            // Badge Text
            ctx.fillStyle = "#000000";
            ctx.fillText(badgeLabel, badgeX + 8, badgeY + 15);
        });

        ctx.restore();
    }

    /**
     * Draws Bird's-Eye View (BEV) Mini-Map Radar in top-right corner
     */
    function drawBEVRadar(roiDistance, speedLimit) {
        if (!latestTelemetry.calibrated) return;

        const bevPoints = latestTelemetry.bev_points || [];
        const radarW = 120;
        const radarH = 180;
        const margin = 15;
        const radarX = canvas.width - radarW - margin;
        const radarY = margin;

        if (radarX < 0 || (radarY + radarH) > canvas.height) return;

        ctx.save();

        // Background Glass Box
        ctx.fillStyle = "rgba(10, 15, 25, 0.88)";
        ctx.fillRect(radarX, radarY, radarW, radarH);
        ctx.strokeStyle = "#00f2fe";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(radarX, radarY, radarW, radarH);

        // Header Title
        ctx.fillStyle = "#00f2fe";
        ctx.font = "bold 11px 'Outfit', sans-serif";
        ctx.fillText(`BEV ${roiDistance.toFixed(0)}m Radar`, radarX + 8, radarY + 16);

        // Grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(radarX, radarY + radarH / 2);
        ctx.lineTo(radarX + radarW, radarY + radarH / 2);
        ctx.moveTo(radarX + radarW / 2, radarY + 22);
        ctx.lineTo(radarX + radarW / 2, radarY + radarH);
        ctx.stroke();

        // Vehicle Radar Blips
        bevPoints.forEach(p => {
            const mx = radarX + Math.max(4, Math.min(radarW - 4, (p.bx / 500) * radarW));
            const my = radarY + 22 + Math.max(4, Math.min(radarH - 26, (p.by / 1000) * (radarH - 22)));
            const blipColor = p.speed > speedLimit ? "#ff3d00" : "#00e676";

            ctx.beginPath();
            ctx.arc(mx, my, 4, 0, 2 * Math.PI);
            ctx.fillStyle = blipColor;
            ctx.shadowColor = blipColor;
            ctx.shadowBlur = 6;
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 9px 'JetBrains Mono', monospace";
            ctx.fillText(`#${p.id}`, mx + 6, my + 3);
        });

        ctx.restore();
    }

    /**
     * Continuous Frame Streaming Loop via WebSocket (Lightweight Downscaled Extraction)
     */
    function startStreaming() {
        function sendFrameLoop() {
            const now = Date.now();
            const minInterval = 33; // 30 FPS for instant fast-vehicle tracking

            if ((now - lastSendTime >= minInterval) && ws && ws.readyState === WebSocket.OPEN && !isSendingFrame) {
                if (video.readyState >= video.HAVE_CURRENT_DATA) {
                    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
                    const jpegQuality = 0.55;
                    const base64Data = captureCanvas.toDataURL("image/jpeg", jpegQuality);

                    const engineVal = engineSelect ? engineSelect.value : "motion";
                    const speedLimitVal = speedLimitInput ? (parseFloat(speedLimitInput.value) || 50.0) : 50.0;
                    const roiDistanceVal = roiDistanceInput ? (parseFloat(roiDistanceInput.value) || 10.0) : 10.0;

                    // Send normalized calibration points (0.0 to 1.0)
                    const normPoints = calibrationPoints.map(pt => [
                        pt[0] / canvas.width,
                        pt[1] / canvas.height
                    ]);

                    const payload = {
                        image: base64Data,
                        points: normPoints,
                        engine: engineVal,
                        speed_limit: speedLimitVal,
                        roi_distance: roiDistanceVal
                    };

                    lastSendTime = now;
                    isSendingFrame = true;
                    ws.send(JSON.stringify(payload));
                }
            }
            setTimeout(sendFrameLoop, 16);
        }
        sendFrameLoop();
    }

    /**
     * Calibration Pointer & Mobile Touch Input Handler
     * Accurately maps screen touch coordinates directly to canvas pixel space
     * Includes debounce protection to guarantee 1 tap = 1 point.
     */
    let lastPointTapTime = 0;

    function handleCalibrationInput(e) {
        if (!isCalibrating) return;
        if (e.cancelable) e.preventDefault();

        const now = Date.now();
        if (now - lastPointTapTime < 200) return; // Debounce duplicate event triggers within 200ms

        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else if (e.changedTouches && e.changedTouches.length > 0) {
            clientX = e.changedTouches[0].clientX;
            clientY = e.changedTouches[0].clientY;
        }

        if (clientX === undefined || clientY === undefined) return;

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const px = Math.max(0, Math.min(canvas.width, Math.round((clientX - rect.left) * scaleX)));
        const py = Math.max(0, Math.min(canvas.height, Math.round((clientY - rect.top) * scaleY)));

        // Guard against duplicate points clicked at the exact same location
        if (calibrationPoints.length > 0) {
            const prev = calibrationPoints[calibrationPoints.length - 1];
            const dist = Math.hypot(px - prev[0], py - prev[1]);
            if (dist < 15) return;
        }

        lastPointTapTime = now;

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

    // Register pointerdown & touchstart on both canvas and wrapper to guarantee mobile touch capture
    canvas.addEventListener("pointerdown", handleCalibrationInput, { passive: false });
    canvas.addEventListener("touchstart", handleCalibrationInput, { passive: false });
    const canvasWrapperElem = document.querySelector(".canvas-wrapper");
    if (canvasWrapperElem) {
        canvasWrapperElem.addEventListener("pointerdown", handleCalibrationInput, { passive: false });
        canvasWrapperElem.addEventListener("touchstart", handleCalibrationInput, { passive: false });
    }

    /**
     * Toggle Calibration Mode
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
     * Reset Calibration Points
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

    function updateCalibrationUI() {
        const count = calibrationPoints.length;
        instructionBanner.textContent = `Click 4 corners on road to calibrate (${count}/4 points set)`;
    }

    if (speedLimitInput && violationLabel) {
        speedLimitInput.addEventListener("input", () => {
            const val = parseFloat(speedLimitInput.value) || 50;
            violationLabel.textContent = `Violations (>${val}km/h)`;
        });
    }

    /**
     * Update Dashboard Sidebar Telemetry & Speed Log
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

        if (topRecords.length > 0) {
            const maxSpeed = Math.max(...topRecords.map(r => r.max_speed));
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
            const isSpeeding = r.is_speeding;
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

    // Start Webcam Feed & App
    initWebcam();
});

