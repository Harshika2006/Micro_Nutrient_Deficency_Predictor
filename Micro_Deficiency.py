import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator
import time 
import joblib

st.set_page_config(page_title="Micros Deficiency",layout='centered')

st.markdown(
    """<style>
    .stApp {
        background-image: url("https://live.staticflickr.com/65535/54641916446_87307d46f6_m.jpg");
        //background-size: cover;
        //background-position: center;
        //background-repeat: repeat;
        //image-rendering: crisp-edges;
        //-webkit-backface-visibility: hidden;
        //-webkit-transform: translate3d(0,0,0);
    }
    Header{
    visibility:hidden
    }
    .block-container{
    background-color:#ffdc7f   ;
    border-radius:30px;
    margin-top:50px;
    padding-top: 10px;
    //padding-bottom: 10px; 
    //margin-bottom:50px;
    border: 5px solid #261b0b;
    }
    .stSelectbox{
        background-color:#ff9a50;
        border: 5px solid #000000 ;
        border-radius: 15px;
        padding: 5px;
        opacity: 0.85;
    }
    
    .stSelectbox label{
    color:#000000
    }
    
    .stButton button{
    background-color: #f29b51;
    color: black;
    border:5px solid #000000 ;
    border-radius: 50px;
    font-weight: bold;
    padding: 0.5em 1.5em;
    opacity: 0.85;
    box-shadow: 2px 2px 4px rgba(112, 123, 124 ,0.1);
    }
      .stButton button:hover {
      border: 5px ridge #004aad ;
      color:#004aad;
        transition: all 0.2s ease;
        opacity: 1:0;
        box-shadow: 2px 2px 4px rgba(112, 123, 124 ,0.1);
      }
      .stAlert {
        background-color: #424949  ;
        color:white;
        //padding: 10px;
        border-radius:30px;
        box-shadow: 2px 2px 4px rgba(112, 123, 124 ,0.1);
    }
     
    </style>""",unsafe_allow_html=True)

df=pd.read_csv("D:/AI/DIET PROJECT/nutrient_diet_sources (2).csv")
vegetarian_foods = dict(zip(df["Nutrient"],df["Veg_Option"]))
non_veg_foods = dict(zip(df["Nutrient"],df["Non_Veg_Option"]))
vegan_foods = dict(zip(df["Nutrient"],df["Vegan_Option"]))

df_sup=pd.read_csv("D:/AI/DIET PROJECT/nutrient_supplements.csv")
Supplement = dict(zip(df_sup["Nutrient"],df_sup["Supplement_with_Dosage"]))
gender_list=['male','female']
age_list=[]

Symptoms1_list=['Slow wound healing' ,'Headache', 'Bone abnormalities', 'Weight gain',
 'Easy bruising', 'Bone pain', 'Impaired taste' ,'Frequent infections',
 'Vision problems', 'Muscle twitches', 'Osteoporosis' ,'Hair loss', 'Numbness'
 'Fatigue' ,'Muscle cramps', 'Low body temperature' ,'Pale skin',
 'Mood changes', 'Excessive bleeding', 'Muscle weakness', 'Tooth decay',
 'Irritability' ,'Neck swelling', 'Seizures', 'Anemia' ,'Weak muscles',
 'Night blindness' ,'Skin issues', 'Dry skin' ,'Heart palpitations',
 'Poor glucose control', 'Bleeding gums', 'Depression' ,'Shortness of breath',
 'Cold sensitivity' ,'Blood clot issues' ,'Poor bone growth',
 'Mental disorders' ,'Confusion', 'Weak immunity', 'Mental fog', 'Weight loss',
 'Tingling in hands', 'Memory issues']

Symptoms2_list=(['Weak immunity', 'Confusion', 'Anemia', 'Cold sensitivity',
'Blood clot issues', 'Weak muscles', 'Slow wound healing', 'Hair loss',
'Night blindness', 'Muscle weakness', 'Numbness', 'Osteoporosis',
'Mental disorders', 'Impaired taste', 'Vision problems',
'Heart palpitations', 'Neck swelling', 'Tooth decay', 'Bone abnormalities',
'Fatigue', 'Poor bone growth', 'Easy bruising', 'Muscle cramps',
'Weight loss', 'Headache', 'Low body temperature', 'Pale skin',
'Shortness of breath', 'Bone pain', 'Frequent infections', 'Dry skin',
'Seizures', 'Skin issues', 'Excessive bleeding', 'Mood changes',
'Irritability', 'Bleeding gums', 'Depression', 'Weight gain',
'Poor glucose control', 'Tingling in hands', 'Mental fog', 'Memory issues',
'Muscle twitches'
])

Symptoms3_list=(['Bleeding gums', 'Low blood pressure', 'Low body temperature',
'Cold sensitivity', 'Excessive bleeding', 'Fatigue', 'Weak immunity',
'Stunted growth', 'Night blindness', 'Numbness', 'Osteoporosis',
'Irregular heartbeat', 'Slow wound healing', 'Vision problems',
'Neck swelling', 'Muscle cramps', 'Bone abnormalities', 'Chest pain',
'Mood changes', 'Anemia', 'Muscle weakness', 'Hair loss', 'Easy bruising',
'Poor glucose control', 'Impaired taste', 'Weight gain', 'Headache',
'Irritability', 'Heavy menstruation', 'Dry skin', 'Thinning hair',
'Coordination issues', 'Skin issues', 'Birth defects (during pregnancy)',
'Mood swings', 'Frequent fractures', 'Bone pain', 'Blood clot issues',
'Tooth loss', 'Weak muscles', 'Seizures', 'Mental retardation', 'Confusion',
'Frequent infections', 'Shortness of breath', 'Muscle twitches',
'Joint pain', 'Heart palpitations', 'Thyroid dysfunction', 'Pale skin',
'Growth delay', 'Depression', 'Weight loss', 'Poor appetite', 'Mental fog',
'Growth problems', 'Poor bone growth', 'Tooth decay', 'Mental disorders',
'Balance problems', 'Tingling in hands', 'Nosebleeds', 'High blood pressure',
'Memory issues', 'Delayed growth'
])

for i in range(10,76):
    age_list.append(i)
@st.cache_data
def translate_list(text_list, target_lang):
    return GoogleTranslator(source='en', target=target_lang).translate_batch(text_list)

@st.cache_data
def translate_single_text(text, target_lang):
    return GoogleTranslator(source='en', target=target_lang).translate(text)

languages=GoogleTranslator().get_supported_languages(as_dict=True)
lang_name={name.title(): code for code, name in languages.items()}
lang_list=[]
lang_keys=lang_name.keys()
for i in lang_keys:
    l=(lang_name.get(i,"unknown"))
    lang_list.append(l)  
lang_select=st.sidebar.selectbox("Language ",lang_list,index=27)
for key, value in lang_name.items():
    if value == lang_select:
        target_lang = key.lower()
        break
if lang_select is not None:
    translated_sym1 = []
    translated_sym2 = []
    translated_sym3 = []
    translated_gender = []
    translated_1=translate_list(Symptoms1_list, target_lang)
    translated_2=translate_list(Symptoms2_list, target_lang)
    translated_3=translate_list(Symptoms3_list, target_lang)
    translated_gen=translate_list(gender_list, target_lang)

    translatedlabel_age=translate_single_text("Age", target_lang)
    translatedlabel_gen=translate_single_text("Gender", target_lang)
    translatedlabel_sym1=translate_single_text("Symptom 1", target_lang)
    translatedlabel_sym2=translate_single_text("Symptom 2", target_lang)
    translatedlabel_sym3=translate_single_text("Symptom 3", target_lang)
    
    title="Micros Nutrients Deficiency Predictor"
    translated_title= translate_single_text(title, target_lang)
    st.markdown(f"""
   <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
    <h1 style="color:#261b0b; font-family:'Pacifico', cursive"; font-size:60px; text-align:center; 
     text-shadow:2px 2px 5px rgba(0,0,0,0.1);'>
     {translated_title}
     </h1>
""", unsafe_allow_html=True)

    v=translate_single_text("Vegetarian food recommendation for", target_lang)
    nv=translate_single_text("Non Vegetarian food recommendation for", target_lang)
    ve=translate_single_text("Vegan food recommendation for", target_lang) 
    su=translate_single_text("Supplement Recommendation", target_lang)
    e1=translate_single_text("Select age , gender and atleast two symptoms", target_lang)
    e2=translate_single_text("Please predict your deficiency before getting recommended food", target_lang)
    r1=translate_single_text("You may be deficient of ", target_lang)
    placeholder1=translate_single_text("Select Your Age", target_lang)
    placeholder2=translate_single_text("Select Your Gender", target_lang)
    placeholder3=translate_single_text("Select First Symptom", target_lang)
    placeholder4=translate_single_text("Select Second Symptom", target_lang)
    placeholder5=translate_single_text("Select Third Symptom", target_lang)
    b1=translate_single_text("Predict Deficiency", target_lang)
    b2=translate_single_text("Recommend food", target_lang)
    b3=translate_single_text("Recommend Supplements", target_lang)
    p1=translate_single_text("Prediction might be wrong", target_lang)
    p2=translate_single_text("Model will work for 2 symptoms but prefer to give three for better results", target_lang)

    col1, col2 = st.columns(2)
    with col1:
        age=st.selectbox(translatedlabel_age,age_list,index=None,placeholder=placeholder1)
    with col2:    
        gen=st.selectbox(translatedlabel_gen,translated_gen,index=None,placeholder=placeholder2)
    s1=st.selectbox(translatedlabel_sym1,translated_1,index=None,placeholder=placeholder3)
    s2=st.selectbox(translatedlabel_sym2,translated_2,index=None,placeholder=placeholder4)
    s3=st.selectbox(translatedlabel_sym3,translated_3,index=None,placeholder=placeholder5)
    if s1 is not None:
        s1_tr=GoogleTranslator(source=target_lang, target='en').translate(s1)  
    if s2 is not None:    
        s2_tr=GoogleTranslator(source=target_lang, target='en').translate(s2)
    if s3 is not None:  
        s3_tr=GoogleTranslator(source=target_lang, target='en').translate(s3)
    if gen is not None:    
        gen_tr=GoogleTranslator(source=target_lang, target='en').translate(gen)
    
    model_1 = joblib.load("defi_model_1.pkl")
    model_2 = joblib.load("defi_model_2.pkl")
    enc_sym1 = joblib.load("encoder_symptom1.pkl")
    enc_sym2 = joblib.load("encoder_symptom2.pkl")
    enc_sym3 = joblib.load("encoder_symptom3.pkl")
    enc_gen = joblib.load("encoder_gender.pkl") 
    enc_def = joblib.load("encoder_deficiency.pkl")    
    if "input1" not in st.session_state:
        st.session_state.input1 = None
    col3, col4, col5= st.columns(3)
    with col3:
        if st.button(b1):
            if (age is not None) and (gen is not None) and (s1 is not None) and (s2 is not None)and (s3 is not None):
                    
                sym1_encoded = enc_sym1.transform([s1_tr])[0]
                sym2_encoded = enc_sym2.transform([s2_tr])[0]
                sym3_encoded = enc_sym3.transform([s3_tr])[0]
                gen_encoded = enc_gen.transform([gen_tr])[0]
                age_value = age
                X2_input = [[age_value,gen_encoded,sym1_encoded,sym2_encoded,sym3_encoded]]
                prediction_2 = model_1.predict(X2_input)[0]
                prediction_label_2 = enc_def.inverse_transform([prediction_2])[0]
                st.session_state.input1=prediction_label_2
                st.markdown(f"""<div style="padding: 1rem; margin-bottom: 1rem;background-color: rgba(30, 100, 50, 0.2); 
        color:rgba(0, 80, 0, 1); 
        border-left: 5px solid rgba(40, 167, 69, 1); 
        border-radius: 5px;">
    <strong>{r1} {prediction_label_2}</strong>
    </div>
    """,
    unsafe_allow_html=True)
   
            elif(age is not None) and (gen is not None) and (s1 is not None) and (s2 is not None) and (s3 is None):
                
                sym1_encoded = enc_sym1.transform([s1])[0]
                sym2_encoded = enc_sym2.transform([s2])[0]
                gen_encoded = enc_gen.transform([gen])[0]
                age_value = age
                X1_input = [[age_value,gen_encoded,sym1_encoded,sym2_encoded]]
                prediction_1 = model_2.predict(X1_input)[0]
                prediction_label_1 = enc_def.inverse_transform([prediction_1])[0]
                st.session_state.input1=prediction_label_1
                st.markdown(f"""<div style="padding: 1rem; margin-bottom: 1rem;background-color: rgba(30, 100, 50, 0.2); 
        color:rgba(0, 80, 0, 1); 
        border-left: 5px solid rgba(40, 167, 69, 1); 
        border-radius: 5px;">
    <strong>{r1} {prediction_label_1}</strong>
    </div>
    """,
    unsafe_allow_html=True)
            else:
                st.error(e1,icon='❌')
                st.stop()    
    with col4:        
        if st.button(b2):
            if st.session_state.input1 is None:
                st.error("⚠️"+ e2)
            else:
                veg_food = vegetarian_foods.get(st.session_state.input1)
                non_veg_food = non_veg_foods.get(st.session_state.input1)
                vegan_food = vegan_foods.get(st.session_state.input1)
                v1=f"{v} {st.session_state.input1}: {veg_food}"
                nv1=f"{nv} {st.session_state.input1}: {non_veg_food}"
                ve1=f"{ve} {st.session_state.input1}: {vegan_food}"
                
                if veg_food:
                    st.markdown(f"""<div style="padding: 1rem;margin-bottom: 1rem;background-color: rgba(0, 90, 110, 0.2);
        color: rgba(0, 38, 66, 1);
        border-left: 5px solid rgba(0, 90, 110, 1);
        border-radius: 5px;">
        ℹ️ <strong>{v1}e : {veg_food}</strong>
    </div>
    """,unsafe_allow_html=True)
                if non_veg_food:
                    st.markdown(f"""<div style="padding: 1rem;margin-bottom: 1rem;background-color: rgba(0, 90, 110, 0.2);
        color: rgba(0, 38, 66, 1);
        border-left: 5px solid rgba(0, 90, 110, 1);
        border-radius: 5px;">
        ℹ️ <strong>{nv1}e : {non_veg_food}</strong>
    </div>
    """,unsafe_allow_html=True)
                if vegan_food:
                    st.markdown(f"""<div style="padding: 1rem;margin-bottom: 1rem;background-color: rgba(0, 90, 110, 0.2);
        color: rgba(0, 38, 66, 1);
        border-left: 5px solid rgba(0, 90, 110, 1);
        border-radius: 5px;">
        ℹ️ <strong>{ve1}e : {vegan_food}</strong>
    </div>
    """,unsafe_allow_html=True)
                      
    with col5:
        if st.button(b3):
            if st.session_state.input1 is None:
                st.error("⚠️"+ e2)
            else:
                nutri_sup = Supplement.get(st.session_state.input1)
                su1=f" {su} {st.session_state.input1}: {nutri_sup}"
                if nutri_sup:
                    st.markdown(f"""<div style="padding: 1rem;margin-bottom: 1rem;background-color: rgba(0, 90, 110, 0.2);
        color: rgba(0, 38, 66, 1);
        border-left: 5px solid rgba(0, 90, 110, 1);
        border-radius: 5px;">
        ℹ️ <strong>{su1}e </strong>
    </div>
    """,unsafe_allow_html=True)
                
st.markdown(f"""<p style = 'color:#2e2e2e'>ℹ️{p1}</p>
""",unsafe_allow_html=True)
st.markdown(f"""<p style = 'color:#2e2e2e'>ℹ️{p2}</p>
""",unsafe_allow_html=True)
st.markdown(
    """
    <div style='position: fixed;
                bottom: 10px;
                right: 10px;
                background-color:#03080b;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 16px;
                z-index: 1000;'>
        © BEN10CODERS \n
        Harshika And Kanishk 
    </div>
    """,
    unsafe_allow_html=True
)                        
