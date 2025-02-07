<?php
session_start();

// Check if there's an error message in the URL parameter
if(isset($_GET['error'])) {
    $errorMessage = $_GET['error'];
    // Display the error message using JavaScript alert
    echo "<script type='text/javascript'>alert('$errorMessage');</script>";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <link rel="stylesheet" href="../css/login.css">
    <script src="https://www.google.com/recaptcha/api.js" async defer></script>
</head>
<body>
    <form action="../actions/login_action.php" method="post">
        <label for="email">Email</label>
        <input type="email" name="email" id="email" placeholder="johndoe@example.com">

        <label for="password">Password</label>
        <input type="password" name="password" id="password"  placeholder="*********">
        
        <div class="g-recaptcha" data-sitekey="6LcdTrgqAAAAAANs1HewfqCeXa5zRGTpoARHQ7qC"></div>

        <button type="submit" name="login">Login</button>
    </form>
    <script src="../js/interaction.js"></script>
</body>
</html>