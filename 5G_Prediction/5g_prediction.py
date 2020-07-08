# importing required libraries
import csv
from typing import List, Any

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.losses import mean_squared_error
from keras.models import Sequential
from matplotlib import pyplot
from numpy import mean
from sklearn.preprocessing import MinMaxScaler


def total_prediction(file_path, col_g, num):
    """

    :param file_path: the path of file read
    :param col_g: the column we need to read from .csv file
    :param num: the row number of the .csv file
    :return: Coefficient of Variation
    """
    # read from csv
    df = pd.read_csv(file_path)
    # sort by row name
    data = df.sort_index(ascending=True, axis=0)
    # creating data frame
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', col_g])
    # copy the 'data' read from .csv file to 'new_data'
    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data[col_g][i] = data[col_g][i]

    # setting index, new_data is the DataFrame type.
    new_data.index = new_data.Date
    # drop the 'Date' column of 'new_data'
    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values
    # the 0th to num th data was included in train set
    train = dataset[0:num-20, :]
    # the size of test set is 12
    test_size = 6
    # to avoid the null prediction, we use the test_size to num of 'dataset' as valid set.
    valid = dataset[test_size:, :]

    # converting dataset into x_train and y_train
    # scaler is used to normalize data within 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    # normalizing data
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(test_size, len(train)):
        x_train.append(scaled_data[i - test_size:i, 0])
        y_train.append(scaled_data[i, 0])
    # convert to numpy
    # the row of x_train is (num-test_size)=64, the col of x_train is test_size=12
    # the row of y_train is test_size=12, the col of x_train is 1
    x_train, y_train = np.array(x_train), np.array(y_train)
    #  tuple, reshape x_train to be 3D.[samples, timesteps, features], (64,12,1)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    # design network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    # x_train: Input data; y_train: Target data; batch_size:Number of samples per gradient update
    history = model.fit(x_train, y_train, epochs=3, batch_size=3, verbose=0)
    # plot loss
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.legend()
    # pyplot.show()

    # predicting with past 'test_size' sample from the train data
    # from len(new_data) - len(valid) - test_size () to end
    inputs = new_data[len(new_data) - len(valid) - test_size:].values
    # reshape into one col
    inputs = inputs.reshape(-1, 1)
    # Scaling features of inputs according to feature_range.
    inputs = scaler.transform(inputs)

    # test the predict result of model, add the data to be predict into x_test list
    x_test = []
    # total_size*test_size, 74*6
    for i in range(test_size, inputs.shape[0]):
        x_test.append(inputs[i - test_size:i, 0])

    # convert to numpy array
    x_test = np.array(x_test)
    # (total_size, test_size, )(74,6,1)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # make prediction
    predict_result = model.predict(x_test)
    # inverse the normalized data to original
    predict_result = scaler.inverse_transform(predict_result)
    # take the first col of result
    predict_result = predict_result[:, 0]
    # the size of predict_result is num - test_size, the last 4 cannot take into consideration
    predict_result = predict_result[0:num - test_size - 4]
    predict_2020 = predict_result[num - test_size - 4:]
    valid = valid[:, 0]
    valid = valid[0:num - test_size - 4]

    # plot the predict and the actual
    pyplot.plot(predict_result, label='predict')
    pyplot.plot(valid, label=col_g)
    pyplot.title(col_g)
    pyplot.legend()
    pyplot.show()

    print(col_g)
    # calculate Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(predict_result, valid))
    # calculate Coefficient of Variation
    cov = rmse / mean(valid)
    print('Coefficient of Variation:', rmse / mean(valid))

    for i in predict_result:
        float(i)
        print(i)
    print("\n")
    return cov,predict_result,valid


def draw_bar(x_index, data_list, xticks, title, x_label, y_label):
    """
    to draw the bar plot
    :param x_index: index
    :param data_list: height
    :param xticks: set the current tick locations labels of the x-axis
    :param title:
    :param x_label:
    :param y_label:
    :return: null
    """
    pyplot.bar(x_index, data_list)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.xticks(x_index, xticks)
    pyplot.title(title)
    pyplot.show()
    pyplot.savefig()


if __name__ == "__main__":
    # index = np.arange(4)
    # cov_rcv = []
    # print('\nTotal prediction:\n')
    # cov_rcv.append(total_prediction('AP.csv', 'total_ap', 76))
    # cov_rcv.append(total_prediction('CALA.csv', 'total_cala', 76))
    # cov_rcv.append(total_prediction('EMEA.csv', 'total_emea', 76))
    # cov_rcv.append(total_prediction('NA.csv', 'total_na', 76))
    # cov_rcv = np.array(cov_rcv)
    # x_ticks = ('total_ap', 'total_cala', 'total_emea', 'total_na')
    # draw_bar(index, cov_rcv, x_ticks, 'Total', 'area', 'CoV')

    predict_result = total_prediction('AP.csv', 'total_ap', 76)[1]
    valid = total_prediction('AP.csv', 'total_ap', 76)[2]
    fig = pyplot.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(predict_result, label='predict')
    ax1.plot(valid, label='total_ap')
    pyplot.legend()
    predict_result=total_prediction('CALA.csv', 'total_cala', 76)[1]
    valid = total_prediction('CALA.csv', 'total_cala', 76)[2]
    ax2 = fig.add_subplot(222)
    ax2.plot(predict_result, label='predict')
    ax2.plot(valid, label='total_cala')
    pyplot.legend()
    predict_result=total_prediction('EMEA.csv', 'total_emea', 76)[1]
    valid = total_prediction('EMEA.csv', 'total_emea', 76)[2]
    ax3 = fig.add_subplot(223)
    ax3.plot(predict_result, label='predict')
    ax3.plot(valid, label='total_emea')
    pyplot.legend()
    predict_result=total_prediction('NA.csv', 'total_na', 76)[1]
    valid = total_prediction('NA.csv', 'total_na', 76)[2]
    ax4 = fig.add_subplot(224)
    ax4.plot(predict_result, label='predict')
    ax4.plot(valid, label='total_na')
    pyplot.legend()
    pyplot.show()


    # print('\n2g prediction:\n')
    # cov_2g = [total_prediction('AP.csv', '2g_ap', 76), total_prediction('CALA.csv', '2g_cala', 76),
    #           total_prediction('EMEA.csv', '2g_emea', 76), total_prediction('NA.csv', '2g_na', 76)]
    # cov_2g = np.array(cov_2g)
    # x_ticks = ('2g_ap', '2g_cala', '2g_emea', '2g_na')
    # draw_bar(index, cov_2g, x_ticks, '2g', 'area', 'CoV')
    #
    # print('\n3g prediction:\n')
    # cov_3g = [total_prediction('AP.csv', '3g_ap', 76), total_prediction('CALA.csv', '3g_cala', 76),
    #           total_prediction('EMEA.csv', '3g_emea', 76), total_prediction('NA.csv', '3g_na', 76)]
    # cov_3g = np.array(cov_3g)
    # x_ticks = ('3g_ap', '3g_cala', '3g_emea', '3g_na')
    # draw_bar(index, cov_3g, x_ticks, '3g', 'area', 'CoV')
    #
    # print('\n4g prediction:\n')
    # cov_4g = [total_prediction('AP.csv', '4g_ap', 76), total_prediction('CALA.csv', '4g_cala', 76),
    #           total_prediction('EMEA.csv', '4g_emea', 76), total_prediction('NA.csv', '4g_na', 76)]
    # cov_4g = np.array(cov_4g)
    # x_ticks = ('4g_ap', '4g_cala', '4g_emea', '4g_na')
    # draw_bar(index, cov_4g, x_ticks, '4g', 'area', 'CoV')

    # total_prediction('AP.csv', '2g_ap', 76)
    # total_prediction('CALA.csv', '2g_cala', 76)
    # total_prediction('EMEA.csv', '2g_emea', 76)
    # total_prediction('NA.csv', '2g_na', 76)
    #
    # total_prediction('AP.csv', '3g_ap', 76)
    # total_prediction('CALA.csv', '3g_cala', 76)
    # total_prediction('EMEA.csv', '3g_emea', 76)
    # total_prediction('NA.csv', '3g_na', 76)
    #
    # total_prediction('AP.csv', '4g_ap', 76)
    # total_prediction('CALA.csv', '4g_cala', 76)
    # total_prediction('EMEA.csv', '4g_emea', 76)
    # total_prediction('NA.csv', '4g_na', 76)
