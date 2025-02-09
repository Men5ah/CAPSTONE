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


<?php
// session_start();
// include_once "../settings/connection.php";
// $conn = get_connection();

// // Google reCAPTCHA API secret key
// $secretKey = "6LcdTrgqAAAAACF2Bg4HBtwymNKdYII9HIWdYbrc";
// $verifyUrl = "https://www.google.com/recaptcha/api/siteverify";

// if (isset($_POST['login'])) {
//     $email = $_POST['email'];
//     $password = $_POST['password'];

//     // Get the reCAPTCHA response token
//     if (isset($_POST['g-recaptcha-response'])) {
//         $recaptchaResponse = $_POST['g-recaptcha-response'];

//         // Verify the CAPTCHA response with Google's API
//         $response = file_get_contents("$verifyUrl?secret=$secretKey&response=$recaptchaResponse");

//         // Decode the JSON response from Google
//         $responseData = json_decode($response);

//         if ($responseData->success) {
//             // CAPTCHA passed
//             $selection = "SELECT * FROM users WHERE email='$email'";
//             $result = mysqli_query($conn, $selection);

//             if (mysqli_num_rows($result) == 0) {
//                 echo "You are not registered with us. Please register if you don't have an account.";
//             } else {
//                 $rows = mysqli_fetch_assoc($result);
//                 if (password_verify($password, $rows['password'])) {
//                     // User authenticated
//                     $_SESSION['email'] = $rows['email'];
//                     $_SESSION['user_id'] = $rows['user_id'];
//                     header("Location: ../views/homepage.php");
//                     exit();
//                 } else {
//                     // Incorrect email or password
//                     $message = "Incorrect email or password! Try again.";
//                     header("Location: ../views/login.php?error=" . urlencode($message));
//                     exit();
//                 }
//             }
//         } else {
//             // CAPTCHA failed
//             $message = "CAPTCHA verification failed. Please try again.";
//             header("Location: ../views/login.php?error=" . urlencode($message));
//             exit();
//         }
//     } else {
//         // No CAPTCHA response received
//         $message = "Please complete the CAPTCHA challenge.";
//         header("Location: ../views/login.php?error=" . urlencode($message));
//         exit();
//     }
// }
?>
<?php
// session_start();
// error_reporting(E_ALL);
// ini_set('display_errors', 1);

// include_once "../settings/connection.php";
// $conn = get_connection();

// if (isset($_POST['login'])) {
//     $email = $_POST['email'];
//     $password = $_POST['password'];

//     if (!isset($_POST['h-captcha-response']) || empty($_POST['h-captcha-response'])) {
//         $message = "hCaptcha response is missing.";
//         header("Location: ../views/login.php?error=" . urlencode($message));
//         exit();
//     }

//     // hCaptcha Verification
//     $hcaptchaResponse = $_POST['h-captcha-response'];
//     $secretKey = "ES_92614266c5224d98930dbe2c5d80993f";
//     $verifyURL = "https://api.hcaptcha.com/siteverify";
//     $data = [
//         'secret' => $secretKey,
//         'response' => $hcaptchaResponse,
//         'remoteip' => $_SERVER['REMOTE_ADDR']
//     ];

//     $options = [
//         'http' => [
//             'header'  => "Content-Type: application/x-www-form-urlencoded\r\n",
//             'method'  => 'POST',
//             'content' => http_build_query($data),
//         ]
//     ];

//     $context  = stream_context_create($options);
//     $result = file_get_contents($verifyURL, false, $context);

//     if ($result === false) {
//         die("hCaptcha verification request failed.");
//     }

//     $captchaSuccess = json_decode($result, true);
//     if (!$captchaSuccess['success']) {
//         die("hCaptcha verification failed. Debug: " . json_encode($captchaSuccess));
//     }

//     // Secure SQL Query
//     $stmt = $conn->prepare("SELECT * FROM users WHERE email = ?");
//     $stmt->bind_param("s", $email);
//     $stmt->execute();
//     $result = $stmt->get_result();

//     if (!$result) {
//         die("SQL Error: " . mysqli_error($conn));
//     }

//     if ($result->num_rows == 0) {
//         $message = "You are not registered with us. Please register.";
//         header("Location: ../views/login.php?error=" . urlencode($message));
//         exit();
//     }

//     $rows = $result->fetch_assoc();

//     if (password_verify($password, $rows['password'])) {
//         $_SESSION['email'] = $rows['email'];
//         $_SESSION['user_id'] = $rows['user_id'];
//         header("Location: ../views/homepage.php");
//         exit();
//     } else {
//         $message = "Incorrect email or password!";
//         header("Location: ../views/login.php?error=" . urlencode($message));
//         exit();
//     }
// }
?>
