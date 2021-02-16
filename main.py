import streamlit as st
import pandas as pd
import joblib
import numpy as np
model = joblib.load('Final_model.pkl')

def predict_model(
        estimator,
        data,
):

    X_test_ = data.copy()
    pred = np.nan_to_num(estimator.predict(X_test_))

    label = pd.DataFrame(pred)
    label.columns = ["Label"]


    try:
        label["Label"] = label["Label"].astype(int)
    except:
        pass


    X_test_ = data.copy()
    X_test_["Label"] = label["Label"].values


    return X_test_

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('Employee.png')
    image_hospital = Image.open('office.jpeg')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if an employee will leave the company')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_hospital)
    st.title("Predicting employee leaving")
    if add_selectbox == 'Online':
        satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
        last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
        number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
        time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
        Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
        promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
        salary = st.selectbox('salary', ['low', 'high','medium'])
        output=""
        input_dict={'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'average_montly_hours':300,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'department':'technical','salary' : salary}
        input_df = pd.DataFrame([input_dict])
        print(input_df)
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

run()
