// let interactionData = {
//     mouseVelocity: [],
//     mouseTrajectory: [],
//     typingSpeedConsistency: [],
//     clickTimingLocation: [],
//     pageDwellTimes: [],
//     scrollFrequency: [],
//     userAgent: navigator.userAgent,
// };

// // Track mouse movement
// document.addEventListener('mousemove', (event) => {
//     const timestamp = Date.now();
//     interactionData.mouseVelocity.push({
//         x: event.movementX,
//         y: event.movementY,
//         timestamp,
//     });
//     interactionData.mouseTrajectory.push({
//         x: event.clientX,
//         y: event.clientY,
//         timestamp,
//     });
// });

// // Track keypress speed
// let typingStart = null;
// document.addEventListener('keydown', () => {
//     const now = Date.now();
//     if (typingStart) {
//         const typingSpeed = now - typingStart;
//         interactionData.typingSpeedConsistency.push(typingSpeed);
//     }
//     typingStart = now;
// });

// // Track click timing and location
// document.addEventListener('click', (event) => {
//     interactionData.clickTimingLocation.push({
//         x: event.clientX,
//         y: event.clientY,
//         timestamp: Date.now(),
//     });
// });

// // Track page dwell time
// const pageEnterTime = Date.now();
// window.addEventListener('beforeunload', () => {
//     const dwellTime = Date.now() - pageEnterTime;
//     interactionData.pageDwellTimes.push(dwellTime);
// });

// // Track scroll frequency
// document.addEventListener('scroll', () => {
//     interactionData.scrollFrequency.push(Date.now());
// });

// // Periodically send data to the server
// setInterval(() => {
//     sendDataToServer();
// }, 5000); // Every 5 seconds

// function sendDataToServer() {
//     fetch('http://localhost/Projects/CAPSTONE/Website/interaction_data/interaction_data.php', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(interactionData),
//     })
//         .then((response) => {
//             if (response.ok) {
//                 console.log('Data successfully sent to server.');
//                 // Reset data after sending
//                 interactionData.mouseVelocity = [];
//                 interactionData.mouseTrajectory = [];
//                 interactionData.typingSpeedConsistency = [];
//                 interactionData.clickTimingLocation = [];
//                 interactionData.pageDwellTimes = [];
//                 interactionData.scrollFrequency = [];
//             } else {
//                 console.error('Error sending data to server.');
//             }
//         })
//         .catch((error) => console.error('Error:', error));
// }
// const sessionData = {
//     session_id: Date.now().toString(), // Unique Session ID
//     mouse_movement: [],
//     typing_patterns: [],
//     click_behaviors: [],
//     scroll_interaction: [],
//     page_dwell_time: 0,
//     time_between_requests: [],
//     user_agent: navigator.userAgent,
//     ip_address: null, // Will be added from the backend
//     timestamp: null,
// };

// let lastActionTime = Date.now(); // Initialize timestamp for request timing

// // Capture mouse movements
// document.addEventListener('mousemove', (event) => {
//     sessionData.mouse_movement.push({ x: event.clientX, y: event.clientY, time: Date.now() });
// });

// // Capture typing patterns
// document.addEventListener('keydown', (event) => {
//     sessionData.typing_patterns.push({ key: event.key, time: Date.now() });
// });

// // Capture click behaviors
// document.addEventListener('click', (event) => {
//     sessionData.click_behaviors.push({ x: event.clientX, y: event.clientY, time: Date.now() });
// });

// // Track scrolling
// let scrollCount = 0;
// window.addEventListener('scroll', () => {
//     scrollCount++;
//     sessionData.scroll_interaction.push({ scrollCount, time: Date.now() });
// });

// // Measure page dwell time
// const startTime = Date.now();
// window.addEventListener('beforeunload', () => {
//     sessionData.page_dwell_time = Date.now() - startTime;
//     sessionData.timestamp = new Date().toISOString();

//     // Send session data to backend
//     sendSessionData(sessionData);
// });

// function sendSessionData(data) {
//     fetch('/collect', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(data),
//     });
// }

// // Unique session ID for each user
// const sessionId = sessionStorage.getItem("sessionId") || generateSessionId();
// sessionStorage.setItem("sessionId", sessionId);

// // Function to generate a random session ID
// function generateSessionId() {
//     return 'sess-' + Math.random().toString(36).substr(2, 9);
// }

// // Data collection object
// const interactionData = {
//     sessionId: sessionId,
//     mouseMovement: [],
//     typingPatterns: [],
//     clickBehaviors: [],
//     pageDwellTime: 0,
//     scrollInteraction: [],
//     timeBetweenRequests: [],
//     userAgent: navigator.userAgent,
//     ipAddress: null
// };

// // Track mouse movement
// document.addEventListener("mousemove", (event) => {
//     interactionData.mouseMovement.push({ x: event.clientX, y: event.clientY, time: Date.now() });
// });

// // Track typing patterns
// document.addEventListener("keydown", (event) => {
//     interactionData.typingPatterns.push({ key: event.key, time: Date.now() });
// });

// // Track clicks
// document.addEventListener("click", (event) => {
//     interactionData.clickBehaviors.push({ x: event.clientX, y: event.clientY, time: Date.now() });
// });

// // Track scroll behavior
// document.addEventListener("scroll", () => {
//     interactionData.scrollInteraction.push({ scrollTop: window.scrollY, time: Date.now() });
// });

// // Calculate dwell time
// let startTime = Date.now();
// window.addEventListener("beforeunload", () => {
//     interactionData.pageDwellTime = Date.now() - startTime;
//     sendData();
// });

// // Time between requests (simulating API calls)
// let lastRequestTime = Date.now();
// function trackRequest() {
//     let now = Date.now();
//     interactionData.timeBetweenRequests.push(now - lastRequestTime);
//     lastRequestTime = now;
// }

// // Fetch user's IP address
// fetch("https://api64.ipify.org?format=json")
//     .then(response => response.json())
//     .then(data => {
//         interactionData.ipAddress = data.ip;
//     });

// // Send data to server
// function sendData() {
//     fetch("../interaction_data/interaction_data.php", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(interactionData)
//     });
// }

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
    fetch("../actions/interaction_data.php", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(interactionData)
    });
}
