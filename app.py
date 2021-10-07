# train ml models directly into streamlit
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using "
"a model built on the Palmer's Penguin's dataset. Use the form below"
" to get started!")
penguin_file = st.file_uploader('Upload your own penguin data')

#when the user has
#yet to upload a file? We can set the default to load our random forest model
#if there is no penguin file

if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()

# if the user has uploaded the file

else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g','sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write('We trained a Random Forest model on these data,'
                 ' it has a score of {}! Use the '
                      'inputs below to try out the model.'.format(score))

#We have now created our model within the app and need to get the inputs from the user
#for our prediction. This time, however, we can make an improvement on what we have
#done before. As of now, each time a user changes an input in our app, the entire Streamlit
#app will rerun. We can use the st.form() and st.submit_form_button()
#functions to wrap the rest of our user inputs in and allow the user to change all of the
#inputs and submit the entire form at once instead of multiple times

with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

# now we have inputs its time to predict
new_prediction = rfc.predict([[bill_length, bill_depth,flipper_length,body_mass, island_biscoe,
                               island_dream,island_torgerson, sex_female,sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write('We predict your penguin is of the {} species'.format(prediction_species))


# Explanation of ML results -> we make a graph in penguins.ml and import in streamlit
# we now import the graph of feature importance in streamlit app

st.write('We used a machine learning (Random Forest) model to '
'predict the species, the features used in this prediction '
' are ranked by relative importance below.')
st.image('feature_importance.png')

#As we can see, bill length, bill depth, and flipper length are the most important variables
#according to our random forest model. A final option for explaining how our model works
#is to plot the distributions of each of these variables by species, and also plot some vertical
#lines representing the user input.

st.write('Below are the histograms for each continuous variable'
'separated by penguin species. The vertical line '
'represents your the inputted value.')

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)
