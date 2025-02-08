<?php
// // Enable error reporting
// error_reporting(E_ALL);
// ini_set('display_errors', 1);

// // Get the raw POST data
// $jsonData = file_get_contents('php://input');
// $data = json_decode($jsonData, true);

// // Validate incoming data
// if (!$data) {
//     http_response_code(400);
//     echo json_encode(['status' => 'error', 'message' => 'Invalid data']);
//     exit();
// }

// // Database connection
// $host = 'localhost';
// $db = 'interactiondb';
// $user = 'root';
// $password = '';

// $conn = new mysqli($host, $user, $password, $db);

// // Check connection
// if ($conn->connect_error) {
//     http_response_code(500);
//     echo json_encode(['status' => 'error', 'message' => 'Database connection failed']);
//     exit();
// }

// // Insert interaction data into the database
// $stmt = $conn->prepare("
//     INSERT INTO interactions (
//         mouse_velocity, 
//         mouse_trajectory, 
//         typing_speed_consistency, 
//         click_timing_location, 
//         page_dwell_times, 
//         scroll_frequency, 
//         user_agent
//     ) VALUES (?, ?, ?, ?, ?, ?, ?)
// ");

// $stmt->bind_param(
//     'sssssss',
//     json_encode($data['mouseVelocity']),
//     json_encode($data['mouseTrajectory']),
//     json_encode($data['typingSpeedConsistency']),
//     json_encode($data['clickTimingLocation']),
//     json_encode($data['pageDwellTimes']),
//     json_encode($data['scrollFrequency']),
//     $data['userAgent']
// );

// if ($stmt->execute()) {
//     echo json_encode(['status' => 'success', 'message' => 'Data saved successfully']);
// } else {
//     http_response_code(500);
//     echo json_encode(['status' => 'error', 'message' => 'Failed to save data']);
// }

// $stmt->close();
// $conn->close();
?>

<?php
// // Enable error reporting for debugging
// error_reporting(E_ALL);
// ini_set('display_errors', 1);

// // Get JSON data from POST request
// $jsonData = file_get_contents('php://input');
// $data = json_decode($jsonData, true);

// // Validate incoming data
// if (!$data || !isset($data['sessionId'])) {
//     http_response_code(400);
//     echo json_encode(['status' => 'error', 'message' => 'Invalid data']);
//     exit();
// }

// // CSV file path
// $csvFile = '../data/interactions.csv';

// // Check if file exists to add headers
// $fileExists = file_exists($csvFile);
// $file = fopen($csvFile, 'a');

// // Add headers if file is newly created
// if (!$fileExists) {
//     fputcsv($file, ['session_id', 'mouse_movement', 'typing_patterns', 'click_behaviors', 'page_dwell_time', 'scroll_interaction', 'time_between_requests', 'user_agent', 'ip_address']);
// }

// // Convert arrays to JSON for storage
// $csvRow = [
//     $data['sessionId'],
//     json_encode($data['mouseMovement']),
//     json_encode($data['typingPatterns']),
//     json_encode($data['clickBehaviors']),
//     $data['pageDwellTime'],
//     json_encode($data['scrollInteraction']),
//     json_encode($data['timeBetweenRequests']),
//     $data['userAgent'],
//     $data['ipAddress']
// ];

// // Write data to CSV
// fputcsv($file, $csvRow);
// fclose($file);

// // Respond with success
// echo json_encode(['status' => 'success', 'message' => 'Data saved successfully']);
?>

<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Get JSON data from POST request
$jsonData = file_get_contents('php://input');
$data = json_decode($jsonData, true);

// Validate incoming data
if (!$data || !isset($data['sessionId'])) {
    http_response_code(400);
    echo json_encode(['status' => 'error', 'message' => 'Invalid data']);
    exit();
}

// CSV file path
$csvFile = '../data/interactions.csv';

// Check if CSV file exists, create it with headers if not
if (!file_exists($csvFile)) {
    $headers = ['session_id', 'mouse_movement', 'typing_patterns', 'click_behaviors', 'page_dwell_time', 'scroll_interaction', 'time_between_requests', 'user_agent', 'ip_address'];
    file_put_contents($csvFile, implode(",", $headers) . "\n");
}

// Convert arrays to JSON before storing
$csvRow = [
    $data['sessionId'],
    json_encode($data['mouseMovement']),
    json_encode($data['typingPatterns']),
    json_encode($data['clickBehaviors']),
    $data['pageDwellTime'],
    json_encode($data['scrollInteraction']),
    json_encode($data['timeBetweenRequests']),
    $data['userAgent'],
    $data['ipAddress']
];

// Append new session data as a single row
file_put_contents($csvFile, implode(",", array_map('addQuotes', $csvRow)) . "\n", FILE_APPEND);

// Function to properly format CSV values
function addQuotes($value) {
    return '"' . str_replace('"', '""', $value) . '"';
}

// Respond with success
echo json_encode(['status' => 'success', 'message' => 'Data saved successfully']);
?>
