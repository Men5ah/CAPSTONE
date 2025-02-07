<?php
// Enable error reporting
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Get the raw POST data
$jsonData = file_get_contents('php://input');
$data = json_decode($jsonData, true);

// Validate incoming data
if (!$data) {
    http_response_code(400);
    echo json_encode(['status' => 'error', 'message' => 'Invalid data']);
    exit();
}

// Database connection
$host = 'localhost';
$db = 'interactiondb';
$user = 'root';
$password = '';

$conn = new mysqli($host, $user, $password, $db);

// Check connection
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(['status' => 'error', 'message' => 'Database connection failed']);
    exit();
}

// Insert interaction data into the database
$stmt = $conn->prepare("
    INSERT INTO interactions (
        mouse_velocity, 
        mouse_trajectory, 
        typing_speed_consistency, 
        click_timing_location, 
        page_dwell_times, 
        scroll_frequency, 
        user_agent
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
");

$stmt->bind_param(
    'sssssss',
    json_encode($data['mouseVelocity']),
    json_encode($data['mouseTrajectory']),
    json_encode($data['typingSpeedConsistency']),
    json_encode($data['clickTimingLocation']),
    json_encode($data['pageDwellTimes']),
    json_encode($data['scrollFrequency']),
    $data['userAgent']
);

if ($stmt->execute()) {
    echo json_encode(['status' => 'success', 'message' => 'Data saved successfully']);
} else {
    http_response_code(500);
    echo json_encode(['status' => 'error', 'message' => 'Failed to save data']);
}

$stmt->close();
$conn->close();
?>
