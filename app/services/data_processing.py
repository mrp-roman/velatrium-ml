def validate_and_sort_data(data):
    # Validate required fields
    required_fields = ["company_id", "system_info", "network_activity"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Sort data by company and category
    sorted_data = {
        "company_id": data["company_id"],
        "system_info": data["system_info"],
        "network_activity": data["network_activity"],
        "timestamp": data.get("timestamp"),
    }

    return sorted_data
