<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>John's Capstone Mockup Website</title>
    <link rel="stylesheet" href="css/register.css">
    <script src="../js/register.js"></script>
</head>
<body>
    <form action="actions/register_action.php" method="post">
        <label for="uname">Username</label>
        <input type="text" name="uname" id="uname" placeholder="John">
        
        <label for="email">Email</label>
        <input type="email" name="email" id="email" placeholder="john.doe@example.com">

        <label for="psswd">Password</label>
        <input type="password" name="psswd" id="psswd">

        <label for="repsswd">Confirm Password</label>
        <input type="password" name="repsswd" id="repsswd">

        <input type="submit" value="Sign Up">
    </form>
</body>
</html>