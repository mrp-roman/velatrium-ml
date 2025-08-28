import csv
from collections import defaultdict
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, Raw

# Define all fields to calculate
FIELDS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

# Initialize data collection and flow tracking
collected_data = []
flows = defaultdict(lambda: {
    "timestamps": [],
    "fwd_lengths": [],
    "bwd_lengths": [],
    "fwd_packets": 0,
    "bwd_packets": 0,
    "flow_start_time": None,
    "flow_end_time": None,
    "destination_port": None,
    "flow_iat": [],
    "fwd_iat": [],
    "bwd_iat": [],
    "fwd_psh_flags": 0,
    "bwd_psh_flags": 0,
    "fwd_urg_flags": 0,
    "bwd_urg_flags": 0,
    "fwd_header_lengths": [],
    "bwd_header_lengths": [],
    "tcp_flags": {  # TCP Flags count
        "FIN": 0,
        "SYN": 0,
        "RST": 0,
        "PSH": 0,
        "ACK": 0,
        "URG": 0,
        "CWE": 0,
        "ECE": 0
    },
    "init_win_bytes_forward": None,
    "init_win_bytes_backward": None,
    "act_data_pkt_fwd": 0,
    "min_seg_size_forward": None,
    "active_intervals": [],
    "last_active_time": None,
    "idle_intervals": [],
    "last_packet_time": None,
    "fwd_bulk_size": 0,
    "bwd_bulk_size": 0,
    "fwd_bulk_count": 0,
    "bwd_bulk_count": 0,
    "fwd_bulk_packets": 0,
    "bwd_bulk_packets": 0,
    "fwd_bulk_rate": 0,
    "bwd_bulk_rate": 0,
})

def process_packet(packet):
    """Process packets and calculate metrics for flows."""
    try:
        # Extract packet details
        src_ip = packet[IP].src if IP in packet else None
        dst_ip = packet[IP].dst if IP in packet else None

        # Enhanced destination port extraction with fallback
        if TCP in packet:
            dst_port = packet[TCP].dport
            header_length = packet[TCP].dataofs * 4  # Header length in bytes
            window_size = packet[TCP].window
            segment_size = len(packet) - header_length
        elif UDP in packet:
            dst_port = packet[UDP].dport
            header_length = 8  # UDP header length
            window_size = None
            segment_size = len(packet) - header_length
        elif Raw in packet and len(packet[Raw].load) >= 2:
            dst_port = int.from_bytes(packet[Raw].load[:2], byteorder="big", signed=False)
            header_length = 0  # Default for raw packets
            window_size = None
            segment_size = len(packet)
        else:
            dst_port = 53
            header_length = 0
            window_size = None
            segment_size = len(packet)

        src_port = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None)
        protocol = "TCP" if TCP in packet else "UDP" if UDP in packet else "Other"
        timestamp = float(packet.time)

        # Define the flow key
        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        reverse_flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)

        # Check if flow exists or needs to be created
        if flow_key in flows:
            flow = flows[flow_key]
        elif reverse_flow_key in flows:
            flow = flows[reverse_flow_key]
        else:
            # Create a new flow
            flows[flow_key] = {
                "timestamps": [timestamp],
                "fwd_lengths": [len(packet)],
                "bwd_lengths": [],
                "fwd_packets": 1,
                "bwd_packets": 0,
                "flow_start_time": timestamp,
                "flow_end_time": timestamp,
                "destination_port": dst_port,
                "flow_iat": [],
                "fwd_iat": [],
                "bwd_iat": [],
                "fwd_psh_flags": 0,
                "bwd_psh_flags": 0,
                "fwd_urg_flags": 0,
                "bwd_urg_flags": 0,
                "fwd_header_lengths": [],
                "bwd_header_lengths": [],
                "tcp_flags": {key: 0 for key in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWE", "ECE"]},
                "init_win_bytes_forward": None,
                "init_win_bytes_backward": None,
                "act_data_pkt_fwd": 0,
                "min_seg_size_forward": None,
                "active_intervals": [],
                "last_active_time": None,
                "idle_intervals": [],
                "last_packet_time": None,
                "fwd_bulk_size": 0,
                "bwd_bulk_size": 0,
                "fwd_bulk_count": 0,
                "bwd_bulk_count": 0,
                "fwd_bulk_packets": 0,
                "bwd_bulk_packets": 0,
                "fwd_bulk_rate": 0,
                "bwd_bulk_rate": 0,
            }
            flow = flows[flow_key]

        # Update forward or backward metrics
        if flow_key in flows:
            # Forward flow
            if len(flow["timestamps"]) > 1:
                flow["fwd_iat"].append(timestamp - flow["timestamps"][-1])
            flow["timestamps"].append(timestamp)
            flow["fwd_lengths"].append(len(packet))
            flow["fwd_packets"] += 1
            flow["fwd_header_lengths"].append(header_length)
            flow["fwd_bulk_packets"] += 1
            flow["fwd_bulk_size"] += len(packet)

            # Calculate idle time
            if flow["last_packet_time"]:
                idle_time = timestamp - flow["last_packet_time"]
                if idle_time > 0:
                    flow["idle_intervals"].append(idle_time)
            flow["last_packet_time"] = timestamp

            # Active intervals
            if flow["last_active_time"]:
                flow["active_intervals"].append(timestamp - flow["last_active_time"])
            flow["last_active_time"] = timestamp

            # Count data packets and update segment size
            if TCP in packet and segment_size > 0:
                flow["act_data_pkt_fwd"] += 1
            if flow["min_seg_size_forward"] is None or segment_size < flow["min_seg_size_forward"]:
                flow["min_seg_size_forward"] = segment_size

            # Set initial forward window size if not already set
            if flow["init_win_bytes_forward"] is None and TCP in packet:
                flow["init_win_bytes_forward"] = window_size

            # Count PSH and URG flags for forward packets
            if TCP in packet:
                flow["fwd_psh_flags"] += 1 if packet[TCP].flags & 0x08 else 0
                flow["fwd_urg_flags"] += 1 if packet[TCP].flags & 0x20 else 0

                # Count TCP flags
                for flag, mask in [("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08),
                                   ("ACK", 0x10), ("URG", 0x20), ("CWE", 0x40), ("ECE", 0x80)]:
                    flow["tcp_flags"][flag] += 1 if packet[TCP].flags & mask else 0

        elif reverse_flow_key in flows:
            # Backward flow
            if len(flow["timestamps"]) > 1:
                flow["bwd_iat"].append(timestamp - flow["timestamps"][-1])
            flow["timestamps"].append(timestamp)
            flow["bwd_lengths"].append(len(packet))
            flow["bwd_packets"] += 1
            flow["bwd_header_lengths"].append(header_length)
            flow["bwd_bulk_packets"] += 1
            flow["bwd_bulk_size"] += len(packet)
            
            # Idle time
            if flow["last_packet_time"]:
                idle_time = timestamp - flow["last_packet_time"]
                if idle_time > 0:
                    flow["idle_intervals"].append(idle_time)
            flow["last_packet_time"] = timestamp

            # Active intervals
            if flow["last_active_time"]:
                flow["active_intervals"].append(timestamp - flow["last_active_time"])
            flow["last_active_time"] = timestamp

            # Set initial backward window size if not already set
            if flow["init_win_bytes_backward"] is None and TCP in packet:
                flow["init_win_bytes_backward"] = window_size

            # Count PSH and URG flags for backward packets
            if TCP in packet:
                flow["bwd_psh_flags"] += 1 if packet[TCP].flags & 0x08 else 0
                flow["bwd_urg_flags"] += 1 if packet[TCP].flags & 0x20 else 0

                # Count TCP flags
                for flag, mask in [("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08),
                                   ("ACK", 0x10), ("URG", 0x20), ("CWE", 0x40), ("ECE", 0x80)]:
                    flow["tcp_flags"][flag] += 1 if packet[TCP].flags & mask else 0

        # Update flow duration
        flow["flow_end_time"] = max(flow["timestamps"]) if flow["timestamps"] else None

    except Exception as e:
        print(f"Error processing packet: {e}")


def calculate_flow_metrics(flow):
    """Calculate all flow metrics from the flow state."""
    timestamps = flow["timestamps"]
    flow_duration = (max(timestamps) - min(timestamps)) if timestamps else 0

    # Forward and backward metrics
    total_fwd_length = sum(flow["fwd_lengths"])
    total_bwd_length = sum(flow["bwd_lengths"])
    total_fwd_packets = len(flow["fwd_lengths"])
    total_bwd_packets = len(flow["bwd_lengths"])

    # Down/Up Ratio
    down_up_ratio = total_fwd_packets / total_bwd_packets if total_bwd_packets > 0 else 0

    # Average Packet Size
    avg_packet_size = (total_fwd_length + total_bwd_length) / (total_fwd_packets + total_bwd_packets) if (total_fwd_packets + total_bwd_packets) > 0 else 0

    # Average Forward/Backward Segment Sizes
    avg_fwd_segment_size = total_fwd_length / total_fwd_packets if total_fwd_packets > 0 else 0
    avg_bwd_segment_size = total_bwd_length / total_bwd_packets if total_bwd_packets > 0 else 0

    # Subflow metrics
    subflow_fwd_bytes = total_fwd_length
    subflow_fwd_packets = total_fwd_packets
    subflow_bwd_packets = total_bwd_packets
    subflow_bwd_bytes = total_bwd_length

    # Forward packet calculations
    fwd_packet_length_max = max(flow["fwd_lengths"], default=0)
    fwd_packet_length_min = min(flow["fwd_lengths"], default=0)
    fwd_packet_length_mean = np.mean(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0
    fwd_packet_length_std = np.std(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0

    # Backward packet calculations
    bwd_packet_length_max = max(flow["bwd_lengths"], default=0)
    bwd_packet_length_min = min(flow["bwd_lengths"], default=0)
    bwd_packet_length_mean = np.mean(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
    bwd_packet_length_std = np.std(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0

    # Flow metrics
    flow_bytes_per_sec = (total_fwd_length + total_bwd_length) / flow_duration if flow_duration > 0 else 0
    flow_packets_per_sec = (total_fwd_packets + total_bwd_packets) / flow_duration if flow_duration > 0 else 0

    # Inter-arrival times (IAT)
    fwd_iat_total = sum(flow["fwd_iat"])
    fwd_iat_mean = np.mean(flow["fwd_iat"]) if flow["fwd_iat"] else 0
    fwd_iat_std = np.std(flow["fwd_iat"]) if flow["fwd_iat"] else 0
    fwd_iat_max = max(flow["fwd_iat"], default=0)
    fwd_iat_min = min(flow["fwd_iat"], default=0)

    bwd_iat_total = sum(flow["bwd_iat"])
    bwd_iat_mean = np.mean(flow["bwd_iat"]) if flow["bwd_iat"] else 0
    bwd_iat_std = np.std(flow["bwd_iat"]) if flow["bwd_iat"] else 0
    bwd_iat_max = max(flow["bwd_iat"], default=0)
    bwd_iat_min = min(flow["bwd_iat"], default=0)

    # Flow Inter-Arrival Times (IAT)
    flow_iat_mean = np.mean(flow["flow_iat"]) if flow["flow_iat"] else 0
    flow_iat_std = np.std(flow["flow_iat"]) if flow["flow_iat"] else 0
    flow_iat_max = max(flow["flow_iat"], default=0)
    flow_iat_min = min(flow["flow_iat"], default=0)

    # Header lengths
    total_fwd_header_length = sum(flow["fwd_header_lengths"])
    total_bwd_header_length = sum(flow["bwd_header_lengths"])

    # Packet rates
    fwd_packets_per_sec = total_fwd_packets / flow_duration if flow_duration > 0 else 0
    bwd_packets_per_sec = total_bwd_packets / flow_duration if flow_duration > 0 else 0

    # Overall packet calculations
    all_packet_lengths = flow["fwd_lengths"] + flow["bwd_lengths"]
    packet_length_min = min(all_packet_lengths, default=0)
    packet_length_max = max(all_packet_lengths, default=0)
    packet_length_mean = np.mean(all_packet_lengths) if all_packet_lengths else 0
    packet_length_std = np.std(all_packet_lengths) if all_packet_lengths else 0
    packet_length_variance = np.var(all_packet_lengths) if all_packet_lengths else 0

    # Active intervals
    active_mean = np.mean(flow["active_intervals"]) if flow["active_intervals"] else 0
    active_std = np.std(flow["active_intervals"]) if flow["active_intervals"] else 0
    active_max = max(flow["active_intervals"], default=0)
    active_min = min(flow["active_intervals"], default=0)

    # Idle intervals
    idle_mean = np.mean(flow["idle_intervals"]) if flow["idle_intervals"] else 0
    idle_std = np.std(flow["idle_intervals"]) if flow["idle_intervals"] else 0
    idle_max = max(flow["idle_intervals"], default=0)
    idle_min = min(flow["idle_intervals"], default=0)

    # Bulk calculations
    fwd_avg_bulk_rate = flow["fwd_bulk_size"] / flow_duration if flow_duration > 0 else 0
    bwd_avg_bulk_rate = flow["bwd_bulk_size"] / flow_duration if flow_duration > 0 else 0
    fwd_avg_bytes_bulk = flow["fwd_bulk_size"] / flow["fwd_bulk_count"] if flow["fwd_bulk_count"] > 0 else 0
    bwd_avg_bytes_bulk = flow["bwd_bulk_size"] / flow["bwd_bulk_count"] if flow["bwd_bulk_count"] > 0 else 0
    fwd_avg_packets_bulk = flow["fwd_bulk_packets"] / flow["fwd_bulk_count"] if flow["fwd_bulk_count"] > 0 else 0
    bwd_avg_packets_bulk = flow["bwd_bulk_packets"] / flow["bwd_bulk_count"] if flow["bwd_bulk_count"] > 0 else 0


    return [
        flow["destination_port"] or 53,  # Destination Port
        flow_duration,
        total_fwd_packets,
        total_bwd_packets,
        total_fwd_length,
        total_bwd_length,
        fwd_packet_length_max,
        fwd_packet_length_min,
        fwd_packet_length_mean,
        fwd_packet_length_std,
        bwd_packet_length_max,
        bwd_packet_length_min,
        bwd_packet_length_mean,
        bwd_packet_length_std,
        flow_bytes_per_sec,
        flow_packets_per_sec, # ADD HERE
        flow_iat_mean,
        flow_iat_std,
        flow_iat_max,
        flow_iat_min,
        fwd_iat_total,
        fwd_iat_mean,
        fwd_iat_std,
        fwd_iat_max,
        fwd_iat_min,
        bwd_iat_total,
        bwd_iat_mean,
        bwd_iat_std,
        bwd_iat_max,
        bwd_iat_min,
        flow["fwd_psh_flags"],                                # Fwd PSH Flags
        flow["bwd_psh_flags"],                                # Bwd PSH Flags
        flow["fwd_urg_flags"],                                # Fwd URG Flags
        flow["bwd_urg_flags"],                             # Bwd URG Flags
        total_fwd_header_length,
        total_bwd_header_length,
        fwd_packets_per_sec,
        bwd_packets_per_sec,
        packet_length_min,
        packet_length_max,
        packet_length_mean,
        packet_length_std,
        packet_length_variance,
        flow["tcp_flags"]["FIN"],  # FIN Flag Count
        flow["tcp_flags"]["SYN"],  # SYN Flag Count
        flow["tcp_flags"]["RST"],  # RST Flag Count
        flow["tcp_flags"]["PSH"],  # PSH Flag Count
        flow["tcp_flags"]["ACK"],  # ACK Flag Count
        flow["tcp_flags"]["URG"],  # URG Flag Count
        flow["tcp_flags"]["CWE"],  # CWE Flag Count
        flow["tcp_flags"]["ECE"],  # ECE Flag Count
        down_up_ratio,
        avg_packet_size,
        avg_fwd_segment_size,
        avg_bwd_segment_size,
        total_fwd_header_length,
        fwd_avg_bytes_bulk,
        fwd_avg_packets_bulk,
        fwd_avg_bulk_rate,
        bwd_avg_bytes_bulk,
        bwd_avg_packets_bulk,
        bwd_avg_bulk_rate,
        subflow_fwd_packets,
        subflow_fwd_bytes,
        subflow_bwd_packets,
        subflow_bwd_bytes,
        flow["init_win_bytes_forward"] or 0,  # Init_Win_bytes_forward
        flow["init_win_bytes_backward"] or 0,  # Init_Win_bytes_backward
        flow["act_data_pkt_fwd"],
        flow["min_seg_size_forward"] or 0,
        active_mean,
        active_std,
        active_max,
        active_min,
        idle_mean,
        idle_std,
        idle_max,
        idle_min,
    ]

def save_to_csv(filename, data):
    """Save collected data to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        writer.writerows(data)

if __name__ == "__main__":
    print("Starting packet capture...")
    sniff(filter="ip", prn=process_packet, timeout=60)  # Capture packets for 60 seconds

    # Process collected flows
    for flow_key, flow in flows.items():
        collected_data.append(calculate_flow_metrics(flow))

    # Save collected data to a CSV file
    save_to_csv("collected_data/network_data.csv", collected_data)
    print("Data collection completed and saved to 'network_data.csv'.")