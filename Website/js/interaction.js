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
    scrollInteraction: [],
    timeBetweenRequests: [],
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
document.addEventListener("scroll", () => {
    interactionData.scrollInteraction.push({ scrollTop: window.scrollY, time: Date.now() });
    sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
});

// Track time between requests
let lastRequestTime = Date.now();
function trackRequest() {
    let now = Date.now();
    interactionData.timeBetweenRequests.push(now - lastRequestTime);
    lastRequestTime = now;
    sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
}

// Track page dwell time
let startTime = Date.now();
window.addEventListener("beforeunload", () => {
    interactionData.pageDwellTime = Date.now() - startTime;
    sendData();
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

// Send Data to Server
function sendData() {
    fetch("../interaction_data/interaction_data.php", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(interactionData)
    });
}
