from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.velatrium

def save_to_mongodb(data):
    # Save sorted data to the appropriate collection
    db.company_data.insert_one(data)

def fetch_sorted_data(company_id):
    # Fetch data for a specific company
    return db.company_data.find_one({"company_id": company_id})
