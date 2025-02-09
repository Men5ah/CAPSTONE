<?php
// define('datasource', 'sql205.infinityfree.com');
// define('db', 'if0_38267486_capstonedb');//databse
// define('uName','if0_38267486');//username
// define('pwd','3nuxX7VdM');//password

define('datasource', 'localhost');
define('db', 'capstonedb');//databse
define('uName','root');//username
define('pwd','');//password

// Function to establish a database connection
function get_connection() {
    $conn = new mysqli(datasource, uName, pwd, db);
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    return $conn;
}

// Function to close the database connection
function close_connection($conn) {
    $conn->close();
}
?>