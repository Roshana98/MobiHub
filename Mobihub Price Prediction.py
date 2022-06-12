from pymongo import MongoClient
import numpy
import matplotlib.pyplot as pyPlot
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# To solve a randomSeed
numpy.random.seed(7)

# Make a connection to MONGODB
client = MongoClient("mongodb+srv://mobihub:mobihub@cluster0.e4sf0.mongodb.net/test?retryWrites=true&w=majority")
mongoDb = client.get_database('MobiHub')
mCollection = mongoDb.get_collection('price_details')

# List of accessible phones in database
availablePhones = ["Huawei Y7"]
#availablePhones = ["Huawei P30 lite"]
#availablePhones = ["Apple iPhone Xs Max"]
#availablePhones = []
# for doc in mCollection.find():
#      name = str(doc['phone_name'])
#      availablePhones.append(name)

print(availablePhones)


def prediction():
    # Creating the dataFrame
    mobihubDataFrame = pandas.DataFrame(allPrices)
    mobihubDataFrame.Date = pandas.to_datetime(mobihubDataFrame.Date)
    mobihubDataFrame = mobihubDataFrame.set_index("Date")

    # Assigning mobihubDataset
    mobihubDataset = mobihubDataFrame.values

    # Illustrate the price history
    pyPlot.figure(figsize=(16, 8))
    pyPlot.title(phoneModel + ' Price History', fontsize=25)
    pyPlot.plot(mobihubDataFrame['Price'])
    pyPlot.xlabel('Date', fontsize=18)
    pyPlot.ylabel('Price', fontsize=18)
    pyPlot.show()

    # Normalizing the mobiuhubDataset that was generated
    mobihubScaler = MinMaxScaler(feature_range=(0, 1))
    mobihubDataset = mobihubScaler.fit_transform(mobihubDataset)

    # mobihubDataset is being split into trainData and testData.
    trainingDatasetSize = int(len(mobihubDataset) * 0.67)
    testingDatasetSize = len(mobihubDataset) - trainingDatasetSize
    trainData = mobihubDataset[0:trainingDatasetSize, :]
    testData = mobihubDataset[trainingDatasetSize:len(mobihubDataset), :]

    # To convert, use NumPy to create a matrix.
    def createNewDataset(newDataset, backStep):
        dataXArray, dataYArray = [], []
        for i in range(len(newDataset) - backStep):
            a = newDataset[i:(i + backStep), 0]
            dataXArray.append(a)
            dataYArray.append(newDataset[i + backStep, 0])
        return numpy.array(dataXArray), numpy.array(dataYArray)

    # Reshaping the x,y data to t and t+1
    backStep = 1
    trainXData, trainYData = createNewDataset(trainData, backStep)
    testXData, testYData = createNewDataset(testData, backStep)

    # Reshaping the input Data [samples, time steps, features]
    trainXData = numpy.reshape(trainXData, (trainXData.shape[0], 1, trainXData.shape[1]))
    testXData = numpy.reshape(testXData, (testXData.shape[0], 1, testXData.shape[1]))

    # Creating the LSTM model and fit the model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, backStep)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainXData, trainYData, epochs=100, batch_size=1, verbose=2)

    # Predicting Train and Test Data
    trainPrediction = model.predict(trainXData)
    testPrediction = model.predict(testXData)

    # Inverting the predicted data
    trainPrediction = mobihubScaler.inverse_transform(trainPrediction)
    trainYData = mobihubScaler.inverse_transform([trainYData])
    testPrediction = mobihubScaler.inverse_transform(testPrediction)
    testYData = mobihubScaler.inverse_transform([testYData])

    # Calculating the  RootMeanSquaredError (RMSE)
    phoneTrainingScore = math.sqrt(mean_squared_error(trainYData[0], trainPrediction[:, 0]))
    print('Train Score of a phone: %.2f RMSE' % phoneTrainingScore)
    phoneTestingScore = math.sqrt(mean_squared_error(testYData[0], testPrediction[:, 0]))
    print('Test Score of a phone: %.2f RMSE' % phoneTestingScore)

    # Shifting the trainData for plotting
    trainPredictionPlot = numpy.empty_like(mobihubDataset)
    trainPredictionPlot[:, :] = numpy.nan
    trainPredictionPlot[backStep:len(trainPrediction) + backStep, :] = trainPrediction

    # Shifting the testData for plotting
    testPredictionPlot = numpy.empty_like(mobihubDataset)
    testPredictionPlot[:, :] = numpy.nan
    testPredictionPlot[len(trainPrediction) + (backStep * 2) - 1:len(mobihubDataset) - 1, :] = testPrediction

    # To Plot the available all data,training and tested data
    pyPlot.figure(figsize=(16, 8))
    pyPlot.title(phoneModel + ' Predicted Price', fontsize=25)
    pyPlot.plot(mobihubScaler.inverse_transform(mobihubDataset), 'b', label='Original Prices')
    pyPlot.plot(trainPredictionPlot, 'r', label='Trained Prices')
    pyPlot.plot(testPredictionPlot, 'g', label='Predicted Prices')
    pyPlot.legend(loc='upper right')
    pyPlot.xlabel('Number of Days', fontsize=18)
    pyPlot.ylabel('Price', fontsize=18)
    pyPlot.show()

    # To PREDICT FUTURE VALUES
    last_month_price = testPrediction[-1]
    last_month_price_scaled = last_month_price / last_month_price
    next_month_price = model.predict(numpy.reshape(last_month_price_scaled, (1, 1, 1)))
    oldPrice = math.trunc(numpy.ndarray.item(last_month_price))
    newPrice = math.trunc(numpy.ndarray.item(last_month_price * next_month_price))
    print("Last Month Price : ", oldPrice)
    print("Next Month Price : ", newPrice)

    # Updating the predicted price in database
    mobileName = mCollection.find_one({'phone_name': phoneModel})
    if bool(mobileName):
        price_update = {
            'predicted_price': newPrice
        }

        mCollection.update_one({'phone_name': phoneModel}, {'$set': price_update})

        print(phoneModel + " PRICE UPDATED")

        # to clear the array
        allPrices.clear()


allPrices = []

# To find the previous prices of a smartphone
for phoneModel in availablePhones:

    for x in mCollection.find({'phone_name': phoneModel}):
        prices = x['prices']
        for y in prices:
            allPrices.append(y)

        prediction()
