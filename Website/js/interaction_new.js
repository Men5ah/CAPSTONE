// // Check for existing session ID or generate a new one
// const sessionId = sessionStorage.getItem("sessionId") || generateSessionId();
// sessionStorage.setItem("sessionId", sessionId);

// function generateSessionId() {
//     return 'sess-' + Math.random().toString(36).substr(2, 9);
// }

// // Cookie Helpers
// function setCookie(name, value, days = 365) {
//     const expires = new Date(Date.now() + days * 864e5).toUTCString();
//     document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
// }

// function getCookie(name) {
//     const value = `; ${document.cookie}`;
//     const parts = value.split(`; ${name}=`);
//     if (parts.length === 2) return decodeURIComponent(parts.pop().split(';').shift());
//     return null;
// }

// function getDurationHistory() {
//     const raw = getCookie("sessionDurations");
//     return raw ? JSON.parse(raw) : [];
// }

// function updateDurationHistory(currentDuration) {
//     let durations = getDurationHistory();
//     durations.push(currentDuration);
//     if (durations.length > 20) durations.shift(); // keep only last 20
//     setCookie("sessionDurations", JSON.stringify(durations));
// }

// function calculateSessionDeviation(currentDuration) {
//     const durations = getDurationHistory();
//     if (durations.length === 0) return 0;
//     const avg = durations.reduce((sum, dur) => sum + dur, 0) / durations.length;
//     return Math.abs(currentDuration - avg);
// }

// // Track start time
// let startTime = Date.now();

// // Interaction data storage
// let interactionData = JSON.parse(sessionStorage.getItem("interactionData") || "{}");
// if (!interactionData.mouseMovement) interactionData.mouseMovement = [];
// if (!interactionData.typingPatterns) interactionData.typingPatterns = [];

// // Fetch user's IP address once
// if (!interactionData.ipAddress) {
//     fetch("https://api64.ipify.org?format=json")
//         .then(response => response.json())
//         .then(data => {
//             interactionData.ipAddress = data.ip;
//             sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
//         });
// }

// // Detect if this is a new device
// function isNewDevice() {
//     const currentDevice = getBrowserType() + navigator.platform;
//     const storedDevice = localStorage.getItem("knownDevice");
//     if (!storedDevice) {
//         localStorage.setItem("knownDevice", currentDevice);
//         return true;
//     }
//     return storedDevice !== currentDevice;
// }

// // Estimate if access is at an unusual time (basic heuristic)
// function isUnusualAccessTime() {
//     const hour = new Date().getHours();
//     return hour >= 22 || hour < 5;
// }

// // Calculate mouse speed (pixels/sec)
// function calculateMouseSpeed() {
//     const movements = interactionData.mouseMovement;
//     if (movements.length < 2) return 0;
//     let totalDistance = 0;
//     let totalTime = (movements[movements.length - 1].time - movements[0].time) / 1000;
//     for (let i = 1; i < movements.length; i++) {
//         let dx = movements[i].x - movements[i - 1].x;
//         let dy = movements[i].y - movements[i - 1].y;
//         totalDistance += Math.sqrt(dx * dx + dy * dy);
//     }
//     return totalTime > 0 ? totalDistance / totalTime : 0;
// }

// // Calculate typing speed (keystrokes/min)
// function calculateTypingSpeed() {
//     const typings = interactionData.typingPatterns;
//     if (typings.length < 2) return 0;
//     let totalTime = (typings[typings.length - 1].time - typings[0].time) / 60000;
//     return totalTime > 0 ? typings.length / totalTime : 0;
// }

// // Day of the week (0 = Sunday)
// function getDayOfWeek() {
//     return new Date().getDay();
// }

// // Time of day label
// function getTimeOfDay() {
//     const hour = new Date().getHours();
//     if (hour >= 5 && hour < 12) return "morning";
//     else if (hour >= 12 && hour < 17) return "afternoon";
//     else if (hour >= 17 && hour < 21) return "evening";
//     else return "night";
// }

// // Identify browser type
// function getBrowserType() {
//     const ua = navigator.userAgent;
//     if (ua.indexOf("Firefox") > -1) return "Firefox";
//     else if (ua.indexOf("Chrome") > -1) return "Chrome";
//     else if (ua.indexOf("Safari") > -1) return "Safari";
//     else if (ua.indexOf("Edge") > -1) return "Edge";
//     else return "Other";
// }

// // Placeholder values (populate from Flask template or API)
// let userId = window.userId || "guest";  // e.g., set server-side
// let loginAttempts = window.loginAttempts || 0;
// let failedLogins = window.failedLogins || 0;

// // Derived values
// let unusualTimeAccess = isUnusualAccessTime();
// let newDeviceLogin = isNewDevice();
// let sessionDurationDeviation = null;  // Will be set on login
// let networkPacketSizeVariance = null; // Not feasible client-side
// let ipRepScore = null; // Get from virustotal API

// // Prepare full payload
// function prepareDataForFlask() {
//     return {
//         user_id: userId,
//         login_attempts: loginAttempts,
//         failed_logins: failedLogins,
//         unusual_time_access: unusualTimeAccess,
//         ip_rep_score: ipRepScore,
//         browser_type: getBrowserType(),
//         new_device_login: newDeviceLogin,
//         session_duration_deviation: sessionDurationDeviation,
//         network_packet_size_variance: networkPacketSizeVariance,
//         mouse_speed: calculateMouseSpeed(),
//         typing_speed: calculateTypingSpeed(),
//         day_of_week: getDayOfWeek(),
//         time_of_day: getTimeOfDay()
//     };
// }

// // Send interaction data to Flask backend
// function sendDataToFlask() {
//     const dataToSend = prepareDataForFlask();
//     return fetch("http://localhost:5001/predict", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(dataToSend)
//     }).catch(err => {
//         console.error("Error sending data to Flask app:", err);
//     });
// }

// // Hook into login form submission to send data before submitting
// document.addEventListener("DOMContentLoaded", () => {
//     const loginForm = document.querySelector("form[action='../actions/login_action.php']");
//     if (loginForm) {
//         loginForm.addEventListener("submit", async (event) => {
//             event.preventDefault();

//             // Calculate session duration deviation before sending
//             const dwellTime = Date.now() - startTime;
//             updateDurationHistory(dwellTime);
//             sessionDurationDeviation = calculateSessionDeviation(dwellTime);

//             try {
//                 await sendDataToFlask();
//             } catch (e) {
//                 console.error("Failed to send interaction data:", e);
//             }

//             // Submit the form after sending data
//             loginForm.submit();
//         });
//     }
// });





//-------------------------------------------------------------------------------
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
            
            // Initialize IP reputation score after getting IP
            initializeIPRepScore(data.ip);
        });
}

// Initialize IP reputation score
async function initializeIPRepScore(ipAddress) {
    try {
        ipRepScore = await getIPReputationScore(ipAddress);
    } catch (e) {
        console.error("Failed to get IP reputation score:", e);
        ipRepScore = 0.15; // Default moderate value (on 0-1 scale)
    }
}

// Estimate network packet size variance (simulation)
function estimateNetworkPacketVariance() {
    // This simulates network packet size variance with a random distribution
    // In real-world scenarios, this would be measured server-side
    const baseVariance = 50; // baseline variance in bytes
    const jitter = Math.random() * 30;
    return baseVariance + jitter;
}

// Function to get IP reputation score from a public API
// Now returns a value between 0 and 1 (0 = low risk, 1 = high risk)
async function getIPReputationScore(ipAddress) {
    try {
        // NOTE: For demonstration, we're using a simple risk assessment based on IP characteristics
        // In production, you should use an actual IP reputation service API
        
        // Calculate a basic score by using some features of the IP
        const ipParts = ipAddress.split('.');
        
        // This is a very simple heuristic - NOT for production use
        // Higher score means higher risk (0 to 1 scale)
        let score = 0;
        
        // If IP starts with certain ranges often used for proxies/VPNs/datacenters
        if (ipParts[0] === '34' || ipParts[0] === '35' || ipParts[0] === '138') {
            score += 0.1;
        }
        
        // Add some randomness for demonstration (0-0.2)
        score += Math.random() * 0.2;
        
        // Ensure score is between 0 and 1
        return Math.min(Math.max(score, 0), 1);
    } catch (error) {
        console.error("Error calculating IP reputation:", error);
        // Return a default moderate value as fallback (on 0-1 scale)
        return 0.15;
    }
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

// Placeholder values (populate from Streamlit app or API)
let userId = window.userId || "guest";  // e.g., set server-side
let loginAttempts = window.loginAttempts || 0;
let failedLogins = window.failedLogins || 0;

// Initialize the previously null values
let sessionDurationDeviation = 0;  // Will be calculated on login
let networkPacketSizeVariance = estimateNetworkPacketVariance(); // Estimated client-side
let ipRepScore = 0.15; // Default value until async fetch completes (0-1 scale)

// Derived values
let unusualTimeAccess = isUnusualAccessTime();
let newDeviceLogin = isNewDevice();

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

// Send interaction data to Streamlit API
function sendDataToFlask() {
    const dataToSend = prepareDataForFlask();
    
    // Convert to the format expected by Streamlit API
    const dataWrapped = {
        "user_id": [dataToSend.user_id],
        "login_attempts": [dataToSend.login_attempts],
        "failed_logins": [dataToSend.failed_logins],
        "unusual_time_access": [dataToSend.unusual_time_access ? 1 : 0],
        "ip_rep_score": [dataToSend.ip_rep_score],
        "browser_type": [dataToSend.browser_type],
        "new_device_login": [dataToSend.new_device_login ? 1 : 0],
        "session_duration_deviation": [dataToSend.session_duration_deviation],
        "network_packet_size_variance": [dataToSend.network_packet_size_variance],
        "mouse_speed": [dataToSend.mouse_speed],
        "typing_speed": [dataToSend.typing_speed],
        "day_of_week": [dataToSend.day_of_week],
        "time_of_day": [dataToSend.time_of_day]
    };
    
    // Use the Streamlit API endpoint instead of Flask
    return fetch("https://mycapstoneapp.streamlit.app/?api&predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataWrapped)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Prediction received:", data);
        // Check if prediction indicates a suspicious login
        if (data && data.length > 0 && data[0].prediction === 1) {
            console.warn("Suspicious login detected!");
            // You can add additional security measures here
        }
        return data;
    })
    .catch(err => {
        console.error("Error sending data to Streamlit app:", err);
    });
}

// Track mouse movements
document.addEventListener("mousemove", function(event) {
    // Throttle recording to avoid excessive data
    if (!interactionData.lastMouseRecord || 
        Date.now() - interactionData.lastMouseRecord > 100) {
        
        interactionData.mouseMovement.push({
            x: event.clientX,
            y: event.clientY,
            time: Date.now()
        });
        
        // Limit stored movement data
        if (interactionData.mouseMovement.length > 100) {
            interactionData.mouseMovement.shift();
        }
        
        interactionData.lastMouseRecord = Date.now();
        sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
    }
});

// Track typing patterns
document.addEventListener("keydown", function(event) {
    // Don't track special keys
    if (event.key.length === 1) {
        interactionData.typingPatterns.push({
            key: event.key,
            time: Date.now()
        });
        
        // Limit stored typing data
        if (interactionData.typingPatterns.length > 100) {
            interactionData.typingPatterns.shift();
        }
        
        sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
    }
});

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
            
            // Update network packet variance estimate
            networkPacketSizeVariance = estimateNetworkPacketVariance();

            try {
                const result = await sendDataToFlask();
                
                // Check if login should be blocked based on prediction
                if (result && result.length > 0 && result[0].prediction === 1) {
                    const confidence = parseFloat(result[0].probability);
                    
                    if (confidence > 0.8) {
                        // High confidence of suspicious activity - block login
                        alert("Suspicious login activity detected. Please contact support.");
                        return; // Prevent form submission
                    } else if (confidence > 0.6) {
                        // Medium confidence - warn but allow
                        if (!confirm("Unusual login activity detected. Continue with login?")) {
                            return; // User chose to cancel
                        }
                    }
                }
                
                // Submit the form after sending data if not blocked
                loginForm.submit();
                
            } catch (e) {
                console.error("Failed to send interaction data:", e);
                // Still submit the form if the security check fails
                loginForm.submit();
            }
        });
    }
});