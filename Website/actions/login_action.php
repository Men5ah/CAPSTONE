<?php
session_start();
include_once "../settings/connection.php";
$conn = get_connection();

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = trim($_POST['email']);
    $password = trim($_POST['password']);

    // Prepare SQL statement
    $stmt = $conn->prepare("SELECT user_id, email, password FROM users WHERE email = ?");
    if (!$stmt) {
        die("SQL Error: " . $conn->error); // Handle SQL error
    }
    
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();

    // Fetch user data
    if ($row = $result->fetch_assoc()) {
        if (password_verify($password, $row['password'])) {
            // Successful login
            $_SESSION['email'] = $row['email'];
            $_SESSION['user_id'] = $row['user_id'];
            
            header("Location: ../views/homepage.php");
            exit();
        }
    }

    // Generic error message to prevent email enumeration
    $_SESSION['error'] = "Invalid email or password!";
    header("Location: ../views/login.php");
    exit();
}
?>
