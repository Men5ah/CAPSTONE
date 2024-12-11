<?php
session_start();
include_once "../settings/connection.php";
$conn = get_connection();

if(isset($_POST['login'])){
    $email = $_POST['email'];
    $password = $_POST['password'];

    $selection = "SELECT * FROM users WHERE email='$email'";

    $result = mysqli_query($conn, $selection);

    if(mysqli_num_rows($result) == 0){
        echo "You are not registered with us. Please register if you don't have an account.";}
    else{
        $rows = mysqli_fetch_assoc($result);
        if(password_verify($password,$rows['password'])){
            $_SESSION['email'] = $rows['email'];
            $_SESSION['user_id'] = $rows['user_id'];
            header("Location: ../views/homepage.php");
            
        }else {
            $message = "Incorrect email or password! Try again";
            header("Location: ../views/login.php?error=" . urlencode($message));
            exit();
        }
    }
}

?>