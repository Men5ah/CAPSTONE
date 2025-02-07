<?php
// session_start();
// include_once "../settings/connection.php";
// $conn = get_connection();

// if(isset($_POST['login'])){
//     $email = $_POST['email'];
//     $password = $_POST['password'];

//     $selection = "SELECT * FROM users WHERE email='$email'";

//     $result = mysqli_query($conn, $selection);

//     if(mysqli_num_rows($result) == 0){
//         echo "You are not registered with us. Please register if you don't have an account.";}
//     else{
//         $rows = mysqli_fetch_assoc($result);
//         if(password_verify($password,$rows['password'])){
//             $_SESSION['email'] = $rows['email'];
//             $_SESSION['user_id'] = $rows['user_id'];
//             header("Location: ../views/homepage.php");
            
//         }else {
//             $message = "Incorrect email or password! Try again";
//             header("Location: ../views/login.php?error=" . urlencode($message));
//             exit();
//         }
//     }
// }
?>

<?php
session_start();
include_once "../settings/connection.php";
$conn = get_connection();

// Google reCAPTCHA API secret key
$secretKey = "6LcdTrgqAAAAACF2Bg4HBtwymNKdYII9HIWdYbrc";
$verifyUrl = "https://www.google.com/recaptcha/api/siteverify";

if (isset($_POST['login'])) {
    $email = $_POST['email'];
    $password = $_POST['password'];

    // Get the reCAPTCHA response token
    if (isset($_POST['g-recaptcha-response'])) {
        $recaptchaResponse = $_POST['g-recaptcha-response'];

        // Verify the CAPTCHA response with Google's API
        $response = file_get_contents("$verifyUrl?secret=$secretKey&response=$recaptchaResponse");

        // Decode the JSON response from Google
        $responseData = json_decode($response);

        if ($responseData->success) {
            // CAPTCHA passed
            $selection = "SELECT * FROM users WHERE email='$email'";
            $result = mysqli_query($conn, $selection);

            if (mysqli_num_rows($result) == 0) {
                echo "You are not registered with us. Please register if you don't have an account.";
            } else {
                $rows = mysqli_fetch_assoc($result);
                if (password_verify($password, $rows['password'])) {
                    // User authenticated
                    $_SESSION['email'] = $rows['email'];
                    $_SESSION['user_id'] = $rows['user_id'];
                    header("Location: ../views/homepage.php");
                    exit();
                } else {
                    // Incorrect email or password
                    $message = "Incorrect email or password! Try again.";
                    header("Location: ../views/login.php?error=" . urlencode($message));
                    exit();
                }
            }
        } else {
            // CAPTCHA failed
            $message = "CAPTCHA verification failed. Please try again.";
            header("Location: ../views/login.php?error=" . urlencode($message));
            exit();
        }
    } else {
        // No CAPTCHA response received
        $message = "Please complete the CAPTCHA challenge.";
        header("Location: ../views/login.php?error=" . urlencode($message));
        exit();
    }
}
?>
