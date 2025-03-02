<?php
include_once "../settings/connection.php";
$conn = get_connection();

// Open the CSV file
$csvFile = fopen("C:/xampp/htdocs/Projects/CAPSTONE/Bot/credentials14.csv", "r");

// Skip the header row (if it exists)
fgetcsv($csvFile);

// Start Transaction
$conn->begin_transaction();
try {
    $stmt = $conn->prepare("INSERT INTO users (username, password, email) VALUES (?, ?, ?)");

    while (($row = fgetcsv($csvFile)) !== FALSE) {
        if (count($row) < 3) continue; // Ensure all fields exist
        
        $uname = trim($row[0]);  // Username
        $email = trim($row[1]);  // Email
        $password = trim($row[2]);  // Password

        // Check for duplicate email
        $check_stmt = $conn->prepare("SELECT COUNT(*) FROM users WHERE email = ?");
        $check_stmt->bind_param("s", $email);
        $check_stmt->execute();
        $check_stmt->bind_result($count);
        $check_stmt->fetch();
        $check_stmt->close();
        
        if ($count > 0) continue;  // Skip duplicates

        // Hash the password using bcrypt
        $security_measure = password_hash($password, PASSWORD_BCRYPT);

        // Insert the record
        $stmt->bind_param("sss", $uname, $security_measure, $email);
        $stmt->execute();
    }

    // Commit transaction
    $conn->commit();
    echo "Users registered successfully!";
} catch (Exception $e) {
    $conn->rollback();
    echo "Error: " . $e->getMessage();
}

// Close connections
fclose($csvFile);
$stmt->close();
$conn->close();
?>