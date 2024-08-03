import pymongo
import re

from bson import ObjectId
class ChatBotMongoDbContext:
    def __init__(self,):
        self.myclient= pymongo.MongoClient("mongodb://root:123456@localhost:27017/ChatgptMongoDbContext?authMechanism=SCRAM-SHA-1&authSource=admin")
        self.ChatgptMongoDbContext = self.myclient["ChatgptMongoDbContext"]
        self.customersCollection = self.ChatgptMongoDbContext["customers"]
        
    def findByAddress(self, address):
        query = {"address": {"$regex": re.compile(address, re.IGNORECASE)}}
        mydoc = self.customersCollection.find(query)
        return mydoc
    def createNewCustomer(self, name, address):
        mydict = { "name": name, "address": address }
        x = self.customersCollection.insert_one(mydict)
        return x
    
    def updateNewCustomer(self, id, name, address):
                
        myquery = { "_id": ObjectId(id) }
        newvalues = { "$set": { "address": address,"name":name } }

        x=self.customersCollection.update_one(myquery, newvalues)
        return x
        
db= ChatBotMongoDbContext()

# print(db.createNewCustomer("du","Van cao"))

print(db.updateNewCustomer("66aba081bed610a67df06b6f","Nguyen Du", "Van Cao, Ba Dinh, Ha noi"))

for x in db.findByAddress("Van cao"):
    print(x)
