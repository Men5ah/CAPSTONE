import random as rd
limit = 10
SSID = 9537
protocol_types = ["TCP", "UDP", "ICMP"]
encryption_used = ["DES", "AES", "None"]
browser_type = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]

for i in range(limit):
    SSID += 1
    packet_size = rd.randrange(50, 1100)
    p_type = rd.choice(protocol_types)
    login_attempts = rd.randrange(1, 16)
    session_duration = rd.uniform(1.0000000000000, 5000.0000000000000)
    e_type = rd.choice(encryption_used)
    ip_reputation = rd.uniform(0.00000000000000000, 1.0000000000000000)
    failed_logins = rd.randrange(0, 5)
    b_type = rd.choice(browser_type)
    unusual_time_access = rd.randint(0, 1)
    attack_detected = rd.randint(0, 1)
    print(f"SID_{SSID},{packet_size},{p_type},{login_attempts},{session_duration},{e_type},{ip_reputation},{failed_logins},{b_type},{unusual_time_access},{attack_detected}")