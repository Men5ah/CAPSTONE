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
