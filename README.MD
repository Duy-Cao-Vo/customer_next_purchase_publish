# STEP 1: EDA & Cleaning Data
### 1.1 Explore Data

data min date:  2019-05-15 00:00:00

data max date:  2020-04-21 00:00:00

Unique customer: 115248

### 1.2 Clean data: customer 1 order, mean order per date > 1.7 and order line > 3
After filtering customer have lable & valid to be create sample for model:  14246

### 1.3 SLIT DATA
![train_test](https://github.com/Duy-Cao-Vo/customer_next_purchase_publish/blob/master/media/train_test_split.png)
# STEP 2: Feature Engineering
### Using LSTM architecture, so have to create feature following time series.
Time step is 5 days, which mean group data in 5 days to one time steps.

This is the most laborious period, our team have to brain-storm different features and testing.

# Step 3: Crate input data
![input_data](https://github.com/Duy-Cao-Vo/customer_next_purchase_publish/blob/master/media/Input_data.PNG)

# Step 4: Trainning model
Model architecture:
![model](https://github.com/Duy-Cao-Vo/customer_next_purchase_publish/blob/master/media/model.PNG)

# Step 5: Validate model
![have_covid-19](https://github.com/Duy-Cao-Vo/customer_next_purchase_publish/blob/master/media/ml_np_6.PNG)