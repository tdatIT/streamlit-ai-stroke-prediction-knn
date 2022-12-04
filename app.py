import dill
import streamlit as st
import encoder_data
import pandas as pd


    
feature=['age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'avg_glucose_level',
 'bmi']


with st.sidebar:
    st.title("Nhập vào thông tin sức khỏe")
    st.image("./img/pixabay_brain-stroke_1200.jpg")
    df = pd.DataFrame.from_dict({
        'age':[float(st.slider("Tuổi",0.080000, 100.000000, 25.0))],
        'hypertension':[st.radio("Tiền sử cao huyết áp",('Yes',"No"))],
        'heart_disease':[st.radio("Tiền sử bệnh tim mạch",('Yes',"No"))],
        'ever_married':[st.radio("Đã kết hôn", ('Yes',"No"))],
        'avg_glucose_level':[st.slider("Chỉ số đường huyết",0.0, 271.740000	, 60.0)],
        'bmi':[st.slider("Chỉ số BMI",14.000000, 48.900000, 20.0)],
})
st.title("Dự Đoán Khả Năng Bệnh Đột Quỵ Bằng AI")

new_data=encoder_data.new_data_num(df)
# st.write(new_data)
with open("knn_model.dill", "rb") as f:
    model = dill.load(f)


# st.write(model.predict(new_data)[0])
st.write("Phần trăm xảy ra đột quỵ: ",model.predict_proba(new_data)[0][1]*100,"%")
# st.write("Probability stroke: ",model.predict_proba(new_data)[0])\\\\\\\\\\\\\
st.image("./img/app-img.png")




