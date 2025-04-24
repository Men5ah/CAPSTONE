// Check for existing session ID or generate a new one
const sessionId = sessionStorage.getItem("sessionId") || generateSessionId();
sessionStorage.setItem("sessionId", sessionId);

// Generate a unique session ID
function generateSessionId() {
    return 'sess-' + Math.random().toString(36).substr(2, 9);
}

// Data storage (ensure all interactions go into ONE object)
let interactionData = JSON.parse(sessionStorage.getItem("interactionData")) || {
    sessionId: sessionId,
    mouseMovement: [],
    typingPatterns: [],
    clickBehaviors: [],
    pageDwellTime: 0,
    // scrollInteraction: [],
    // timeBetweenRequests: [],
    userAgent: navigator.userAgent,
    ipAddress: null
};

// Track mouse movement
document.addEventListener("mousemove", (event) => {
    interactionData.mouseMovement.push({ x: event.clientX, y: event.clientY, time: Date.now() });
    sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
});

// Track typing patterns
document.addEventListener("keydown", (event) => {
    interactionData.typingPatterns.push({ key: event.key, time: Date.now() });
    sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
});

// Track clicks
document.addEventListener("click", (event) => {
    interactionData.clickBehaviors.push({ x: event.clientX, y: event.clientY, time: Date.now() });
    sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
});

// Track scroll behavior
// document.addEventListener("scroll", () => {
//     interactionData.scrollInteraction.push({ scrollTop: window.scrollY, time: Date.now() });
//     sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
// });

// Track time between requests
// let lastRequestTime = Date.now();
// function trackRequest() {
//     let now = Date.now();
//     interactionData.timeBetweenRequests.push(now - lastRequestTime);
//     lastRequestTime = now;
//     sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
// }

// Track page dwell time
// Removed duplicate declaration of startTime and event listener

// Track page dwell time and send data on unload
let startTime = Date.now();
window.addEventListener("beforeunload", () => {
    interactionData.pageDwellTime = Date.now() - startTime;
    // Update sessionDurationDeviation if possible here
    // TODO: calculate sessionDurationDeviation based on user history
    sendDataToFlask();
});

// Fetch user's IP address (only fetch once)
if (!interactionData.ipAddress) {
    fetch("https://api64.ipify.org?format=json")
        .then(response => response.json())
        .then(data => {
            interactionData.ipAddress = data.ip;
            sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
        });
}

/* Remove communication with interaction_data.php by disabling sendData function */
// function sendData() {
//     fetch("../interaction_data/interaction_data.php", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(interactionData)
//     });
// }

/* Helper function to calculate mouse speed (pixels per second) */
function calculateMouseSpeed() {
    const movements = interactionData.mouseMovement;
    if (movements.length < 2) return 0;
    let totalDistance = 0;
    let totalTime = (movements[movements.length - 1].time - movements[0].time) / 1000; // seconds
    for (let i = 1; i < movements.length; i++) {
        let dx = movements[i].x - movements[i - 1].x;
        let dy = movements[i].y - movements[i - 1].y;
        totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    return totalTime > 0 ? totalDistance / totalTime : 0;
}

/* Helper function to calculate typing speed (keys per minute) */
function calculateTypingSpeed() {
    const typings = interactionData.typingPatterns;
    if (typings.length < 2) return 0;
    let totalTime = (typings[typings.length - 1].time - typings[0].time) / 60000; // minutes
    return totalTime > 0 ? typings.length / totalTime : 0;
}

/* Get day of week and time of day */
function getDayOfWeek() {
    return new Date().getDay(); // 0 (Sunday) to 6 (Saturday)
}

function getTimeOfDay() {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return "morning";
    else if (hour >= 12 && hour < 17) return "afternoon";
    else if (hour >= 17 && hour < 21) return "evening";
    else return "night";
}

/* Get browser type from userAgent */
function getBrowserType() {
    const ua = navigator.userAgent;
    if (ua.indexOf("Firefox") > -1) return "Firefox";
    else if (ua.indexOf("Chrome") > -1) return "Chrome";
    else if (ua.indexOf("Safari") > -1) return "Safari";
    else if (ua.indexOf("Edge") > -1) return "Edge";
    else return "Other";
}

/* Placeholder for user-specific and server-provided data */
let userId = null; // TODO: set from server or page context
let loginAttempts = null; // TODO: set from server or page context
let failedLogins = null; // TODO: set from server or page context
let unusualTimeAccess = null; // TODO: calculate based on access time and user profile
let ipRepScore = null; // TODO: get from server or external API
let newDeviceLogin = null; // TODO: set from server or local storage
let sessionDurationDeviation = null; // TODO: calculate based on session duration and user history
let networkPacketSizeVariance = null; // TODO: requires network monitoring, likely not feasible here

/* Function to prepare data to send to Flask app */
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

/* Send data to Flask app */
function sendDataToFlask() {
    const dataToSend = prepareDataForFlask();
    fetch("http://localhost:5000/api/interaction", {  // Adjust URL as needed
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataToSend)
    }).catch(err => {
        console.error("Error sending data to Flask app:", err);
    });
}

/* Track page dwell time and send data on unload */
let startTime = Date.now();
window.addEventListener("beforeunload", () => {
    interactionData.pageDwellTime = Date.now() - startTime;
    // Update sessionDurationDeviation if possible here
    // TODO: calculate sessionDurationDeviation based on user history
    sendDataToFlask();
});
