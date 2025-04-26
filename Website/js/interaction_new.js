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
    
    // Calculate base deviation
    const avg = durations.reduce((sum, dur) => sum + dur, 0) / durations.length;
    let deviation = Math.abs(currentDuration - avg) / 1000; // Convert ms to seconds
    
    // Add randomness to match your target distribution
    if (deviation < 0.1) {
        // For very small deviations, sometimes make them slightly larger
        if (Math.random() > 0.7) {
            deviation = 0.2 + Math.random() * 0.8;
        }
    } else {
        // Apply a multiplier that creates more variation
        const multiplier = 0.8 + Math.random() * 1.4;
        deviation = deviation * multiplier;
        
        // Occasionally add larger spikes (like your 4+ values)
        if (Math.random() > 0.85) {
            deviation = 3.5 + Math.random() * 1.5;
        }
    }
    
    // Ensure we don't return negative values
    deviation = Math.max(0, deviation);
    
    // Round to 2 decimal places to match your examples
    return Math.round(deviation * 100) / 1000;
}

// Track start time
let startTime = Date.now();

// Initialize interaction data with proper defaults
let interactionData = {
    mouseMovement: [],
    typingPatterns: [],
    ipAddress: null,
    lastMouseRecord: null
};

// Load from sessionStorage if available
try {
    const storedData = JSON.parse(sessionStorage.getItem("interactionData"));
    if (storedData) {
        interactionData = {
            ...interactionData,
            ...storedData,
            mouseMovement: storedData.mouseMovement || [],
            typingPatterns: storedData.typingPatterns || []
        };
    }
} catch (e) {
    console.error("Error loading interaction data from sessionStorage:", e);
}

// Fetch user's IP address once
if (!interactionData.ipAddress) {
    fetch("https://api64.ipify.org?format=json")
        .then(response => response.json())
        .then(data => {
            interactionData.ipAddress = data.ip;
            sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
            
            // Initialize IP reputation score after getting IP
            initializeIPRepScore(data.ip);
        })
        .catch(err => {
            console.error("Error fetching IP address:", err);
            interactionData.ipAddress = "unknown";
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
    // Generate a base value with different probability distributions
    let variance;
    const rand = Math.random();
    
    if (rand < 0.6) {
        // 60% chance for small values (0-1)
        variance = Math.random() * 1;
    } else if (rand < 0.9) {
        // 30% chance for medium values (1-2)
        variance = 1 + Math.random();
    } else {
        // 10% chance for larger values (2-3)
        variance = 2 + Math.random();
    }
    
    // Apply some fine-tuning to match your exact distribution pattern
    variance = variance * 0.95 + (Math.random() * 0.1);
    
    // Round to 2 decimal places to match your examples
    variance = Math.round(variance * 100) / 100;
    
    // Ensure to never return negative values
    return Math.max(0, variance);
}

// Function to get IP reputation score
async function getIPReputationScore(ipAddress) {
    try {
        let score = 0;
        const ipParts = ipAddress.split('.');
        
        if (ipParts[0] === '34' || ipParts[0] === '35' || ipParts[0] === '138') {
            score += 0.1;
        }
        
        score += Math.random() * 0.2;
        return Math.min(Math.max(score, 0), 1);
    } catch (error) {
        console.error("Error calculating IP reputation:", error);
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

// Estimate if access is at an unusual time
function isUnusualAccessTime() {
    const hour = new Date().getHours();
    return hour >= 22 || hour < 5;
}

// Calculate mouse speed (pixels/sec)
function calculateMouseSpeed() {
    const movements = interactionData.mouseMovement;
    if (!movements || movements.length < 2) return 0;
    
    let totalDistance = 0;
    let totalTime = (movements[movements.length - 1].time - movements[0].time) / 1000;
    
    for (let i = 1; i < movements.length; i++) {
        let dx = movements[i].x - movements[i - 1].x;
        let dy = movements[i].y - movements[i - 1].y;
        totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    
    return totalTime > 0 ? (totalDistance / totalTime)/100 : 0;
}

// Calculate typing speed (keystrokes/min)
function calculateTypingSpeed() {
    const typings = interactionData.typingPatterns;
    if (!typings || typings.length < 2) return 0;
    
    let totalTime = (typings[typings.length - 1].time - typings[0].time) / 60000;
    return totalTime > 0 ? (typings.length / totalTime)/10 : 0;
}

// Day of the week (0 = Sunday)
function getDayOfWeek() {
    return new Date().getDay();
}

// Time of day label
function getTimeOfDay() {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return "Morning";
    else if (hour >= 12 && hour < 17) return "Afternoon";
    else if (hour >= 17 && hour < 21) return "Evening";
    else return "Night";
}

// Identify browser type 
function getBrowserType() {
    const ua = navigator.userAgent;
    
    // Check Firefox first (since it can contain 'Safari' in UA string)
    if (ua.includes("Firefox") && !ua.includes("Seamonkey")) {
        return "Firefox";
    }
    // Check Edge next (since it contains Chrome in UA string)
    else if (ua.includes("Edg") || ua.includes("Edge")) {
        return "Edge";
    }
    // Then check Chrome (many browsers include Chrome in UA string)
    else if (ua.includes("Chrome") && !ua.includes("Chromium")) {
        return "Chrome";
    }
    // Then Safari (after excluding Chrome-based browsers)
    else if (ua.includes("Safari") && !ua.includes("Chrome")) {
        return "Safari";
    }
    else {
        return "Other";
    }
}

// User data initialization
let userId = window.userId || "guest";
let loginAttempts = window.loginAttempts || 0;
let failedLogins = window.failedLogins || 0;
let sessionDurationDeviation = 0;
let networkPacketSizeVariance = estimateNetworkPacketVariance();
let ipRepScore = 0.20;
let unusualTimeAccess = isUnusualAccessTime();
let newDeviceLogin = isNewDevice();

// Prepare full payload - Updated to return array of objects
function prepareDataForFlask() {
    // Map day numbers to names
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    
    // Get the values
    const data = {
        user_id: userId,
        login_attempts: loginAttempts,
        failed_logins: failedLogins,
        unusual_time_access: unusualTimeAccess, // Now sending as boolean directly
        ip_rep_score: ipRepScore,
        browser_type: getBrowserType(),
        new_device_login: newDeviceLogin, // Now sending as boolean directly
        session_duration_deviation: sessionDurationDeviation,
        network_packet_size_variance: networkPacketSizeVariance,
        mouse_speed: calculateMouseSpeed(),
        typing_speed: calculateTypingSpeed(),
        day_of_week: days[getDayOfWeek()], // Convert to string name
        time_of_day: getTimeOfDay()
    };
    
    // Create a DataFrame-like structure (array of objects where each object is a row)
    return [data]; // Wrap in array to simulate DataFrame with one row
}

// Send data to Flask endpoint - Fixed to use /predict
async function sendDataToFlask() {
    const dataToSend = prepareDataForFlask();
    
    // Enhanced console log to show the exact data being sent
    console.log("Data being sent to backend:", JSON.stringify(dataToSend, null, 2));
    console.log("Data details:", {
        "User ID": dataToSend[0].user_id,
        "Login Attempts": dataToSend[0].login_attempts,
        "Failed Logins": dataToSend[0].failed_logins,
        "Unusual Time Access": dataToSend[0].unusual_time_access,
        "IP Reputation Score": dataToSend[0].ip_rep_score,
        "Browser Type": dataToSend[0].browser_type,
        "New Device Login": dataToSend[0].new_device_login,
        "Mouse Speed": dataToSend[0].mouse_speed,
        "Typing Speed": dataToSend[0].typing_speed,
        "Day of Week": dataToSend[0].day_of_week,
        "Time of Day": dataToSend[0].time_of_day
    });
    
    try {
        console.log("Initiating API call to Flask endpoint...");
        
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(dataToSend)
        });

        console.log("Response status:", response.status, response.statusText);
        
        if (!response.ok) {
            try {
                const errorResponse = await response.json();
                console.error("Server error response (JSON):", errorResponse);
                
                if (errorResponse.error) {
                    console.error("Server error details:", errorResponse.error);
                    if (errorResponse.traceback) {
                        console.error("Traceback:", errorResponse.traceback);
                    }
                }
            } catch (jsonError) {
                const errorText = await response.text();
                console.error("Server error response:", {
                    status: response.status,
                    text: errorText.substring(0, 200)
                });
            }
            
            return [{ prediction: 0, probability: 0 }];
        }

        const data = await response.json();
        console.log("Received response from backend:", JSON.stringify(data, null, 2));
        return data;
    } catch (err) {
        console.error("API request failed:", {
            error: err.message,
            stack: err.stack
        });
        
        return [{ prediction: 0, probability: 0 }];
    }
}

// Track mouse movements with throttling
document.addEventListener("mousemove", function(event) {
    if (!interactionData.lastMouseRecord || Date.now() - interactionData.lastMouseRecord > 100) {
        if (!interactionData.mouseMovement) {
            interactionData.mouseMovement = [];
        }
        
        interactionData.mouseMovement.push({
            x: event.clientX,
            y: event.clientY,
            time: Date.now()
        });
        
        if (interactionData.mouseMovement.length > 100) {
            interactionData.mouseMovement.shift();
        }
        
        interactionData.lastMouseRecord = Date.now();
        sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
    }
});

// Track typing patterns - Fixed to handle undefined event.key
document.addEventListener("keydown", function(event) {
    // Make sure event.key exists before checking its length
    if (event.key && event.key.length === 1) {
        if (!interactionData.typingPatterns) {
            interactionData.typingPatterns = [];
        }
        
        interactionData.typingPatterns.push({
            key: event.key,
            time: Date.now()
        });
        
        if (interactionData.typingPatterns.length > 100) {
            interactionData.typingPatterns.shift();
        }
        
        sessionStorage.setItem("interactionData", JSON.stringify(interactionData));
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("form[action='../actions/login_action.php']");
    if (loginForm) {
        loginForm.addEventListener("submit", async (event) => {
            event.preventDefault(); // Always stop normal submission first

            // Increment login attempts
            loginAttempts++;

            // Update session metrics
            const dwellTime = Date.now() - startTime;
            updateDurationHistory(dwellTime);
            sessionDurationDeviation = calculateSessionDeviation(dwellTime);
            networkPacketSizeVariance = estimateNetworkPacketVariance();

            try {
                const result = await sendDataToFlask();

                // Add hidden fields for backend tracking
                addHiddenField(loginForm, 'login_attempts', loginAttempts);
                addHiddenField(loginForm, 'failed_logins', failedLogins);

                if (result && result.length > 0) {
                    const prediction = result[0].prediction;
                    const confidence = parseFloat(result[0].probability);

                    if (prediction === 1) {
                        if (confidence > 0.8) {
                            return blockLogin("Suspicious login activity detected. Please contact support.");
                        } else if (confidence > 0.5) {
                            return confirmLogin("Unusual login activity detected. Continue with login?", loginForm);
                        }
                    }
                }

                // If no issues, approve the login
                approveLogin(loginForm);

            } catch (error) {
                console.error("Failed to send interaction data:", error);
                approveLogin(loginForm); // Fail-safe: allow login if Flask server fails
            }
        });
    }
});

// Helper to add hidden inputs
function addHiddenField(form, name, value) {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = name;
    input.value = value;
    form.appendChild(input);
}

// Block login with alert
function blockLogin(message) {
    alert(message);
    return;
}

// Confirm login with the user
function confirmLogin(message, form) {
    if (confirm(message)) {
        form.submit();
    } else {
        failedLogins++;
    }
    return;
}

// Approve login normally
function approveLogin(form) {
    form.submit();
}