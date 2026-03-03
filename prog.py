import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(layout="wide")


# 1. МАЪЛУМОТЛАРНИ ЮКЛАШ ВА МОДЕЛНИ ТАЙЁРЛАШ
@st.cache_data
def load_data():
    df = pd.read_excel('CTS_Processed_Data.xlsx')
    target = 'Ultrasound_CTS (Yes/No)'
    # Тозалаш ва кодлаш
    df_clean = df.copy()
    df_clean[target] = df_clean[target].map({'Yes': 1, 'No': 0})
    return df, df_clean

raw_df, clean_df = load_data()

target_col = 'Ultrasound_CTS (Yes/No)' # Faylingizdagi joylashuvi bo'yicha
import joblib
import os
if os.path.exists("kts_model.pkl"):
    model = joblib.load("kts_model.pkl")
else:
    df = clean_df.dropna(subset=[target_col])
    features_to_drop = [target_col, 'Patient ID']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    from sklearn.preprocessing import LabelEncoder
    for col in ['Sex', 'Left/Right' ]:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "kts_model.pkl")


# --- SIDEBAR MENU ---
# st.sidebar.title("ККС Эксперт Тизими")
# menu = st.sidebar.radio("Меню:", ["Беморлар рўйхати", "Статистик таҳлил", "Янги ташхис"])

st.sidebar.title("Меню")
if st.sidebar.button("👥 Беморлар рўйхати", use_container_width=True):
    st.session_state.menu = "Беморлар рўйхати"
if st.sidebar.button("📊 Статистик таҳлил", use_container_width=True):
    st.session_state.menu = "Статистик таҳлил"
if st.sidebar.button("⚡ Янги ташхис", use_container_width=True):
    st.session_state.menu = "Янги ташхис"

if 'menu' not in st.session_state:
    st.session_state.menu = "Беморлар рўйхати"
menu = st.session_state.menu

translations = {
       'Sex':"Жинс", 'Age': "Ёш", 'Left/Right':"Чап/Ўнг",
       'Ultrasound_Mid-Forearm Thickness (mm)': "билак ўрта қисмидаги нервнинг қалинлиги",
       'Ultrasound_Mid-Forearm Mediolateral Diameter (mm)': "билак ўрта қисмидаги нервнинг медиолатерал диаметри",
       'Ultrasound_Mid-Forearm Maximum Cross-Sectional Area (mm²)': "билак ўрта қисмидаги нервнинг максимал кўндаланг кесим юзаси",
       'Ultrasound_Pisiform Level Thickness (mm)': "нўхатсимон суякдаги нервнинг қалинлиги",
       'Ultrasound_Pisiform Level Mediolateral Diameter (mm)': "нўхатсимон суякдаги нервнинг медиолатерал диаметри",
       'Ultrasound_Pisiform Level Maximum Cross-Sectional Area (mm²)': "нўхатсимон суякдаги нервнинг максимал кўндаланг кесим юзаси",
       'Ultrasound_Hamate Level Thickness (mm)': "илгаксимон суякдаги нервнинг қалинлиги",
       'Ultrasound_Hamate Level Mediolateral Diameter (mm)': "илгаксимон суякдаги нервнинг медиолатерал диаметри",
       'Ultrasound_Hamate Level Maximum Cross-Sectional Area (mm²)': "илгаксимон суякдаги нервнинг максимал кўндаланг кесим юзаси",
       'Ultrasound_Carpal Tunnel Narrowest Thickness (mm)': "Кaрпал туннелнинг энг тор қисмидаги нервнинг қалинлиги",
    #    'Ultrasound_CTS (Yes/No)': "КТС",
       'Wrist–Elbow Conduction Velocity (m/s)': "билак – тирсак сегментида ўтказувчанлик тезлиги (m/s)",
       'Wrist–Elbow Latency (ms)': "билак – тирсак сегментида латентлик даври (ms)",
       'Wrist–Elbow Evoked Potential (mV)': "билак – тирсак сегментида чақирилган потенциал амплитудаси (mV)",
       'Middle Finger–Wrist Conduction Velocity (m/s)': "ўрта бармоқ – билак сегментида ўтказувчанлик тезлиги (m/s)",
       'Middle Finger–Wrist Latency (ms)': "ўрта бармоқ – билак сегментида латентлик даври (ms)",
       'Middle Finger–Wrist Evoked Potential (mV)': "ўрта бармоқ – билак сегментида чақирилган потенциал амплитудаси (mV)",
       'Index Finger–Wrist Conduction Velocity (m/s)': "кўрсаткич бармоқ – билак сегментида ўтказувчанлик тезлиги (m/s)",
       'Index Finger–Wrist Latency (ms)': "кўрсаткич бармоқ – билак сегментида латентлик даври (ms)",
       'Index Finger–Wrist Evoked Potential (mV)': "кўрсаткич бармоқ – билак сегментида чақирилган потенциал амплитудаси (mV)",
       'Thumb–Wrist Conduction Velocity (m/s)': "бош бармоқ – билак сегментида ўтказувчанлик тезлиги (m/s)",
       'Thumb–Wrist Latency (ms)': "бош бармоқ – билак сегментида латентлик даври (ms)",
       'Thumb–Wrist Evoked Potential (mV)': "бош бармоқ – билак сегментида чақирилган потенциал амплитудаси (mV)",
}

raw_df=raw_df.rename(columns=translations)

if menu == "Беморлар рўйхати":
    st.header("Беморлар маълумотлар базаси")
    
    st.subheader("Умумий жадвал")
    st.dataframe(raw_df)
    
    st.subheader("Индивидуал кўздан кечириш")
    patient_ids = raw_df['Patient ID'].unique()
    selected_id = st.selectbox("Бемор ID рақамини танланг:", patient_ids)
    
    patient_data = raw_df[raw_df['Patient ID'] == selected_id]
    st.table(patient_data)

elif menu == "Статистик таҳлил":
    st.header("📊 Статистик ва Информатив таҳлил")
    
    st.subheader("Гуруҳлар бўйича асосий кўрсаткичлар")
    temp_df = clean_df.copy()   
    temp_df = temp_df.rename(columns=translations) 

    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if 'Patient ID' in numeric_cols:
        numeric_cols.remove('Patient ID')

    # min-max-mean-median statistikasi
    agg_translations = {
        'min': 'Минимал',
        'max': 'Максимал',
        'mean': 'Ўртача',
        'median': 'Медиана'
    }
    
    temp_df[target_col] = temp_df[target_col].map({1: "КТС бор", 0: "КТС йўқ"})
    temp_df = temp_df.drop(columns=['Patient ID'], errors='ignore').rename(columns=translations)
    stats_df = temp_df.groupby(target_col)[numeric_cols].agg(['min', 'max', 'mean', 'median'])
    stats_df.index.names = ['Белгилар']
    stats_formatted = stats_df.stack(level=0).swaplevel(0, 1).sort_index()
    stats_final = stats_formatted.unstack(level=0).T.swaplevel(0,1).sort_index()
    # st.dataframe(stats_df.T, use_container_width=True,
    #              column_config={
    #                 "Белгилар": st.column_config.Column(width="large"),
    #                 "Кўрсаткич": st.column_config.Column(width="medium"),
    #                 "КТС бор": st.column_config.NumberColumn(width="small", format="%.3f"),
    #                 "КТС йўқ": st.column_config.NumberColumn(width="small", format="%.3f"),
    # })
    res = stats_df.copy().T
    res['Фарқ (%)'] = ((res['КТС бор'] - res['КТС йўқ'])/res['КТС йўқ'] * 100)
    res.index = res.index.set_levels([res.index.levels[0], [agg_translations[i] for i in res.index.levels[1]]])
    styled_res = res.style.format({
        'КТС бор': "{:.3f}",
        'КТС йўқ': "{:.3f}",
        'Фарқ (%)': "{:+.2f}%"
    }).set_table_styles([
        {'selector': '', 'props': [('display', 'block'), 
                                ('max-height', '400px'), 
                                ('overflow-y', 'auto'), 
                                ('border', '1px solid #ddd')]},
        {'selector': 'thead th', 'props': [('position', 'sticky'), 
                                        ('top', '0'), 
                                        ('z-index', '2'), 
                                        ('background-color', '#f4f4f4'), 
                                        ('border-bottom', '2px solid #ccc')]},
        {'selector': 'tbody th', 'props': [('background-color', '#f9f9f9'), 
                                        ('border', '1px solid #ddd')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), 
                                    ('padding', '8px'), 
                                    ('text-align', 'center')]},
        {'selector': 'th', 'props': [('padding', '8px'), 
                                    ('text-align', 'center')]}
    ])
    

    st.write(styled_res.to_html(), unsafe_allow_html=True)


    # Correlation Heatmap
    st.subheader("")
    st.subheader("Белгилар орасидаги корреляция")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(temp_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


elif menu == "Янги ташхис":
    st.header("Янги бемор учун AI-диагностика ва тактика")
    model_features = list(translations.keys())
    if 'Ultrasound_CTS (Yes/No)' in model_features:
        model_features.remove('Ultrasound_CTS (Yes/No)')

    st.subheader("Бемор кўрсаткичларини киритинг")
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    test_data = {
        'Sex': 'Female', 'Age': 32, 'Left/Right': 'Left',
        'Ultrasound_Mid-Forearm Thickness (mm)': 1.7,
        'Ultrasound_Mid-Forearm Mediolateral Diameter (mm)': 3.3,
        'Ultrasound_Mid-Forearm Maximum Cross-Sectional Area (mm²)': 4.0,
        'Ultrasound_Pisiform Level Thickness (mm)': 2.1,
        'Ultrasound_Pisiform Level Mediolateral Diameter (mm)': 5.6,
        'Ultrasound_Pisiform Level Maximum Cross-Sectional Area (mm²)': 10.0,
        'Ultrasound_Hamate Level Thickness (mm)': 1.8,
        'Ultrasound_Hamate Level Mediolateral Diameter (mm)': 5.6,
        'Ultrasound_Hamate Level Maximum Cross-Sectional Area (mm²)': 8.0,
        'Ultrasound_Carpal Tunnel Narrowest Thickness (mm)': 2.0,
        'Wrist–Elbow Conduction Velocity (m/s)': 57.5,
        'Wrist–Elbow Latency (ms)': 3.1,
        'Wrist–Elbow Evoked Potential (mV)': 4.6,
        'Middle Finger–Wrist Conduction Velocity (m/s)': 54.0,
        'Middle Finger–Wrist Latency (ms)': 2.5,
        'Middle Finger–Wrist Evoked Potential (mV)': 23.0,
        'Index Finger–Wrist Conduction Velocity (m/s)': 54.0,
        'Index Finger–Wrist Latency (ms)': 2.5,
        'Index Finger–Wrist Evoked Potential (mV)': 21.0,
        'Thumb–Wrist Conduction Velocity (m/s)': 54.1,
        'Thumb–Wrist Latency (ms)': 1.94,
        'Thumb–Wrist Evoked Potential (mV)': 25.0
    }

    if st.button("Тест бемор маълумотларини тўлдириш"):
        st.session_state.temp_input = test_data
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    if 'temp_input' not in st.session_state:
        st.session_state.temp_input = {k: (0.0 if isinstance(v, float) else v) for k, v in test_data.items()}

    for i, (orig_name, trans_name) in enumerate(translations.items()):
        if orig_name == 'Ultrasound_CTS (Yes/No)': continue

        target_col = [col1, col2, col3][i % 3]
        current_val = st.session_state.temp_input.get(orig_name)

        with target_col:
            if orig_name == 'Sex':
                options = ["Male", "Female"]
                val = st.selectbox(trans_name, options, index=options.index(current_val) if current_val in options else 0)
                input_values[orig_name] = 1 if val == "Male" else 0
            elif orig_name == 'Left/Right':
                options = ["Right", "Left"]
                val = st.selectbox(trans_name, options, index=options.index(current_val) if current_val in options else 0)
                input_values[orig_name] = 1 if val == "Right" else 0
            else:
                input_values[orig_name] = st.number_input(trans_name, value=float(current_val), format="%.2f", key=f"in_{orig_name}")
    st.markdown("""
        <style>
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            vertical-align: top !important;
        }
        .stNumberInput label, .stSelectbox label {
            min-height: 80px !important; /* Ҳаммасида бир хил баландлик */
            margin-bottom: 0px !important;
            padding-bottom: 5px !important;
            display: flex;
            align-items: flex-end; /* Матнларни пастки чизиқ бўйича текислайди */
        }
        div[data-baseweb="input"], div[data-baseweb="select"] {
            margin-top: 0px !important;
        }
        [data-testid="stVerticalBlock"] > div {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button("Таҳлилни бошлаш"):
        input_array = np.array([input_values[f] for f in model_features]).reshape(1, -1)
        prob = model.predict_proba(input_array)[0][1]
        
        st.divider()
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("ККС эҳтимоли", f"{prob*100:.1f}%")
            if prob > 0.5:
                st.error("Ташхис: КТС тасдиқланади")
            else:
                st.success("Ташхис: Соғлом")

        st.subheader("📊 Норматив таҳлил (халқаро стандартлар билан қиёслаш)")
        
        WORLD_STANDARDS = {
            'Ultrasound_Pisiform Level Maximum Cross-Sectional Area (mm²)': {'norm': 10.0, 'unit': 'mm²', 'type': 'max'},
            'Wrist–Elbow Latency (ms)': {'norm': 3.7, 'unit': 'ms', 'type': 'max'},
            'Middle Finger–Wrist Latency (ms)': {'norm': 3.2, 'unit': 'ms', 'type': 'max'},
            'Index Finger–Wrist Latency (ms)': {'norm': 3.2, 'unit': 'ms', 'type': 'max'},
            'Ultrasound_Carpal Tunnel Narrowest Thickness (mm)': {'norm': 2.0, 'unit': 'mm', 'type': 'max'}
        }
        
        comparison_data = []
        for i, (feat, std) in enumerate(WORLD_STANDARDS.items(), 1):
            user_val = input_values[feat]
            norm_val = std['norm']
            diff_pct = ((user_val - norm_val) / norm_val) * 100
            comparison_data.append({
                "№": i,
                "Кўрсаткич": translations[feat],
                "Беморда": f"{user_val:.2f} {std['unit']}",
                "Жаҳон нормаси": f"{norm_val:.2f} {std['unit']}",
                "Фарқ (%)": f"{diff_pct:+.1f}%"
            })
        
        df_comp = pd.DataFrame(comparison_data).set_index("№")
        st.table(df_comp)

        st.subheader("🩺 Тавсия этилган даволаш тактикаси")
        
        # --- ТАКТИКАНИ 5 ТА ДАРАЖАГА БЎЛИШ (Jahon Standarti - Bland Classification асосида) ---
        csa = input_values['Ultrasound_Pisiform Level Maximum Cross-Sectional Area (mm²)']
        m_latency = input_values['Middle Finger–Wrist Latency (ms)']
        i_latency = input_values['Index Finger–Wrist Latency (ms)']
        we_latency = input_values['Wrist–Elbow Latency (ms)']
        thickness = input_values['Ultrasound_Carpal Tunnel Narrowest Thickness (mm)']

        score = 0
        if csa >= 15.0: score += 3
        elif csa >= 12.0: score += 1
        
        if m_latency >= 4.5 or i_latency >= 4.5: score += 3
        elif m_latency > 3.2 or i_latency > 3.2: score += 1
        
        if we_latency > 4.2: score += 2
        if thickness < 1.6: score += 1
        if prob > 0.85: score += 2
        elif prob > 0.5: score += 1

        if score >= 8:
            st.error("### 5-ДАРАЖА: ЖУДА ОҒИР")
            st.write("**Ҳолат:** Нерв ўтказувчанлиги деярли йўқолган, мушаклар атрофияси хавфи юқори.")
            st.write("**Тактика:** Шошилинч жарроҳлик амалиёти ва узоқ муддатли реабилитация.")
        
        elif 6 <= score < 8:
            st.error("### 4-ДАРАЖА: ОҒИР")
            st.write("**Ҳолат:** Аксонал зарарланиш ва кучли демиелинизация.")
            st.write("**Тактика:** Жарроҳлик декомпрессияси тавсия этилади. Консерватив даво фойда бермаслиги мумкин.")
            
        elif 4 <= score < 6:
            st.warning("### 3-ДАРАЖА: ЎРТА")
            st.write("**Ҳолат:** Нерв шиши (CSA) ва мотор латентликнинг сезиларли узайиши.")
            st.write("**Тактика:** Консерватив даво: Стериод инъекциялар, тунги шина (splinting) ва физиотерапия.")
            
        elif 2 <= score < 4:
            st.info("### 2-ДАРАЖА: ЕНГИЛ")
            st.write("**Ҳолат:** Фақат сезги толаларида импульс секинлашган. Мотор функция сақланган.")
            st.write("**Тактика:** Шиналаш, витаминотерапия ва иш жойи эргономикасини ўзгартириш.")
            
        else:
            st.success("### 1-ДАРАЖА: ЖУДА ЕНГИЛ / НОРМА")
            st.write("**Ҳолат:** Кўрсаткичлар нормал ёки минимал функционал ўзгаришлар.")
            st.write("**Тактика:** Профилактик машқлар ва 6-12 ойдан кейин назорат кўриги.")

