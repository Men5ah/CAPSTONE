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
    <!-- <script src="https://www.google.com/recaptcha/api.js" async defer></script> -->
    <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
</head>
<body>
    <form action="../actions/login_action.php" method="post">
        <label for="email">Email</label>
        <input type="email" name="email" id="email" placeholder="johndoe@example.com">

        <label for="password">Password</label>
        <input type="password" name="password" id="password"  placeholder="*********">
        
        <div class="h-captcha" data-sitekey="d83a798a-d1f8-49e8-b5de-47221902b622"></div>

        <button type="submit" name="login">Login</button>
    </form>
    <script src="../js/interaction.js"></script>
</body>
</html>