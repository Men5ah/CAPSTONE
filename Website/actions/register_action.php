<?php
include_once "../settings/connection.php";
$conn = get_connection();

if($_SERVER["REQUEST_METHOD"]=="POST") {
    $uname = mysqli_real_escape_string($conn, $_POST['uname']);
    $email = mysqli_real_escape_string($conn, $_POST['email']);
    $password = mysqli_real_escape_string($conn, $_POST['psswd']);
    $security_measure = password_hash($password, PASSWORD_BCRYPT);;

    $insert = "INSERT INTO users(username, password, email) values ('$uname','$security_measure','$email')";

    $result = mysqli_query($conn, $insert);
    
    if($result) {
        header( "location: ../views/login.php");
        exit();
    } else {
        echo "Error: ". $conn->error;
    }
}
?>