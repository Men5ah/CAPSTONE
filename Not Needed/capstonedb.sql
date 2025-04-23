-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Feb 08, 2025 at 01:00 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `capstonedb`
--

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `user_id` bigint(20) UNSIGNED NOT NULL,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `email` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`user_id`, `username`, `password`, `created_at`, `email`) VALUES
(1, 'qidotycava', '$2y$10$cxjZsIwskjKXI4/DomiToOYORJEdMgnUMZrbEVwmPN3S.qNgQYu.y', '2024-12-11 02:05:50', 'tizaka@mailinator.com'),
(2, 'miqowef', '$2y$10$YLmHRk66.Oi5bB4hJDEl3e4uFPx.ZznbTyDVUvYI4smef6lHx7jh.', '2024-12-11 02:06:38', 'fatarozix@mailinator.com'),
(3, 'Men5ah', '$2y$10$FxmUgOB2HlVIfmZ6M001I..rI8p8iMSkvP6yqZ0iw8MS7KShh7nnq', '2024-12-11 02:10:37', 'john@example.com'),
(4, 'anthony777', '$2y$10$mza7DmXBYQkqwXDxXx/XMuRFElujf5MmQ7zei2dubyc.UwFXkcNMS', '2024-12-11 02:35:20', 'anthony@example.com'),
(5, 'kwamina5', '$2y$10$r0jwS548nxCvGgxLivtxtOcAD0BPBQD9kXKj2FHVswai.Xww5PVmO', '2024-12-11 02:42:07', 'kwamina@example.com');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `user_id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
