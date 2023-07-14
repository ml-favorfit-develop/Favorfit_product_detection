from pymongo import MongoClient
from bson import ObjectId

class DBAdmin:
    def __init__(self, DB_HOST, DB_NAME, COLLECTION_NAME):
        self.client = MongoClient(DB_HOST)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        print(f"Dataset -> host={DB_HOST}, name={DB_NAME}")
    
    def select_document(self, db_id):
        doc = self.collection.find_one({"_id":db_id})
        return doc
    
    def delete_field(self, db_id, field_name):
        self.collection.update_one({"_id": db_id}, {'$unset': {field_name: None}})

    def put_field(self, db_id, field_name, contents=None):
        self.collection.update_one({"_id": ObjectId(db_id)}, {'$set': {field_name: contents}})

    def push_field_contents(self, db_id, field_name, content):
        self.collection.update_one({"_id": db_id}, {'$push': {field_name: content}})

    def pull_field_contents(self, db_id, field_name, content):
        self.collection.update_one({"_id": db_id}, {'$pull': {field_name: content}})

    def close(self):
        self.client.close()
        print(f"{self.db} DB closed")
