// Check for existing session ID or generate a new one
const sessionId = sessionStorage.getItem("sessionId") || generateSessionId();
sessionStorage.setItem("sessionId", sessionId);

function generateSessionId() {
    return 'sess-' + Math.random().toString(36).substr(2, 9);
}

// Cookie Helpers
function setCookie(name, value, days = 365) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return decodeURIComponent(parts.pop().split(';').shift());
    return null;
}

function getDurationHistory() {
    const raw = getCookie("sessionDurations");
    return raw ? JSON.parse(raw) : [];
}

function updateDurationHistory(currentDuration) {
    let durations = getDurationHistory();
    durations.push(currentDuration);
    if (durations.length > 20) durations.shift(); // keep only last 20
    setCookie("sessionDurations", JSON.stringify(durations));
}

function calculateSessionDeviation(currentDuration) {
    const durations = getDurationHistory();
    if (durations.length === 0) return 0;
    const avg = durations.reduce((sum, dur) => sum + dur, 0) / durations.length;
    return Math.abs(currentDuration - avg);
}

// Track start time
let startTime = Date.now();

// Interaction data storage
let interactionData = JSON.parse(sessionStorage.getItem("interactionData") || "{}");
if (!interactionData.mouseMovement) interactionData.mouseMovement = [];
if (!interactionData.typingPatterns) interactionData.typingPatterns = [];

// Fetch user's IP address once
if (!interactionData.ipAddress) {
    fetch("https://api64.ipify.org?format=json")
        .then(response => response.json())
        .then(data => {
            interactionData.ipAddress = data.ip;
            sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
        });
}

// Detect if this is a new device
function isNewDevice() {
    const currentDevice = getBrowserType() + navigator.platform;
    const storedDevice = localStorage.getItem("knownDevice");
    if (!storedDevice) {
        localStorage.setItem("knownDevice", currentDevice);
        return true;
    }
    return storedDevice !== currentDevice;
}

// Estimate if access is at an unusual time (basic heuristic)
function isUnusualAccessTime() {
    const hour = new Date().getHours();
    return hour >= 22 || hour < 5;
}

// Calculate mouse speed (pixels/sec)
function calculateMouseSpeed() {
    const movements = interactionData.mouseMovement;
    if (movements.length < 2) return 0;
    let totalDistance = 0;
    let totalTime = (movements[movements.length - 1].time - movements[0].time) / 1000;
    for (let i = 1; i < movements.length; i++) {
        let dx = movements[i].x - movements[i - 1].x;
        let dy = movements[i].y - movements[i - 1].y;
        totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    return totalTime > 0 ? totalDistance / totalTime : 0;
}

// Calculate typing speed (keystrokes/min)
function calculateTypingSpeed() {
    const typings = interactionData.typingPatterns;
    if (typings.length < 2) return 0;
    let totalTime = (typings[typings.length - 1].time - typings[0].time) / 60000;
    return totalTime > 0 ? typings.length / totalTime : 0;
}

// Day of the week (0 = Sunday)
function getDayOfWeek() {
    return new Date().getDay();
}

// Time of day label
function getTimeOfDay() {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return "morning";
    else if (hour >= 12 && hour < 17) return "afternoon";
    else if (hour >= 17 && hour < 21) return "evening";
    else return "night";
}

// Identify browser type
function getBrowserType() {
    const ua = navigator.userAgent;
    if (ua.indexOf("Firefox") > -1) return "Firefox";
    else if (ua.indexOf("Chrome") > -1) return "Chrome";
    else if (ua.indexOf("Safari") > -1) return "Safari";
    else if (ua.indexOf("Edge") > -1) return "Edge";
    else return "Other";
}

// Placeholder values (populate from Flask template or API)
let userId = window.userId || "guest";  // e.g., set server-side
let loginAttempts = window.loginAttempts || 0;
let failedLogins = window.failedLogins || 0;

// Derived values
let unusualTimeAccess = isUnusualAccessTime();
let newDeviceLogin = isNewDevice();
let sessionDurationDeviation = null;  // Will be set on login
let networkPacketSizeVariance = null; // Not feasible client-side
let ipRepScore = null; // Optional: Get from backend using interactionData.ipAddress

// Prepare full payload
function prepareDataForFlask() {
    return {
        user_id: userId,
        login_attempts: loginAttempts,
        failed_logins: failedLogins,
        unusual_time_access: unusualTimeAccess,
        ip_rep_score: ipRepScore,
        browser_type: getBrowserType(),
        new_device_login: newDeviceLogin,
        session_duration_deviation: sessionDurationDeviation,
        network_packet_size_variance: networkPacketSizeVariance,
        mouse_speed: calculateMouseSpeed(),
        typing_speed: calculateTypingSpeed(),
        day_of_week: getDayOfWeek(),
        time_of_day: getTimeOfDay()
    };
}

// Send interaction data to Flask backend
function sendDataToFlask() {
    const dataToSend = prepareDataForFlask();
    return fetch("http://localhost:5001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataToSend)
    }).catch(err => {
        console.error("Error sending data to Flask app:", err);
    });
}

// Hook into login form submission to send data before submitting
document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("form[action='../actions/login_action.php']");
    if (loginForm) {
        loginForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            // Calculate session duration deviation before sending
            const dwellTime = Date.now() - startTime;
            updateDurationHistory(dwellTime);
            sessionDurationDeviation = calculateSessionDeviation(dwellTime);

            try {
                await sendDataToFlask();
            } catch (e) {
                console.error("Failed to send interaction data:", e);
            }

            // Submit the form after sending data
            loginForm.submit();
        });
    }
});
