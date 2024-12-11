<?php
session_start();

function checkLogin(){
    if(!isset($_SESSION['userID'])){
        header("Location: ../views/login.php");
        die();
    }else{
        return true;
    }
}
?>