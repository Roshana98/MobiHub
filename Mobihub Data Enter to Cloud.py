from pymongo import MongoClient
import pandas


client = MongoClient(
    "mongodb+srv://mobihub:mobihub@cluster0.e4sf0.mongodb.net/test?retryWrites=true&w=majority")
mongoDb = client.get_database('MobiHub')

ans = input("""
    ENTER 1 TO ADD PRICE CSV
    ENTER 2 TO ADD DETAILS CSV
    """)
if ans == "1":
    while True:
        # ADD PHONE PRICES OBJECT
        phoneModel = input("Enter the csv file name : ")
        readingCsv = phoneModel + ".csv"

        dataFrame = pandas.read_csv(readingCsv, encoding="ISO-8859-1")
        records_ = dataFrame.to_dict(orient='records')

        # ADD PRICES ARRAY
        # dateArray = []
        # priceArray = []
        # for doc in records_:
        #     date = str(doc['Date'])
        #     price = doc['Price']
        #     dateArray.append(date)
        #     priceArray.append(price)
        #
        # new = {
        #     'phone_name': phoneModel,
        #     'dates': dateArray,
        #     'prices': priceArray,
        #     'predicted_price': 0.0
        # }
        # mCollection = mongoDb.get_collection('phone_prices')
        # result = mCollection.insert_one(new)

        new = {
            'phone_name': phoneModel,
            'prices': records_,
            'predicted_price': 0
        }
        mCollection = mongoDb.get_collection('price_details')
        result = mCollection.insert_one(new)

        print("ADDED")

elif ans == "2":

    # ADD PHONE DETAILS
    file = input("Enter the csv file name : ")
    readingCsv = file + ".csv"

    dataFrame = pandas.read_csv(readingCsv, encoding="ISO-8859-1")
    records_ = dataFrame.to_dict(orient='records')

    mCollection = mongoDb.get_collection('phone_details')
    result = mCollection.insert_many(records_)

    print("ADDED")

else:
    print("\n Not Valid Choice Try again")
