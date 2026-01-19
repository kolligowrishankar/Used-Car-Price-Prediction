import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image, ImageDraw
import joblib
model = joblib.load('model_rf.pkl') 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# --- 1. LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('C:\ML Project\German cars\germany_cars.csv')
    except:
        df = pd.read_csv('germany_cars.csv')
    
    # Clean up names to match filenames (removes spaces, etc.)
    df['make'] = df['make'].astype(str).str.strip()
    df['model'] = df['model'].astype(str).str.strip()
    return df

@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

df = load_data()
pipe = load_model()

# --- 2. AUTOMATIC IMAGE FINDER ---
def get_car_image(make_name):
    """
    Automatically looks for an image file that matches the car name.
    Example: If make_name is 'BMW', it looks for 'car_logos/BMW.png'
    """
    folder_path = "C:\ML Project\German cars\car_images"  # The folder name
    
    # It checks 3 common formats automatically
    valid_extensions = [".png", ".jpg", ".jpeg"]
    
    for ext in valid_extensions:
        # Construct the path: car_logos/BMW.png
        file_path = os.path.join(folder_path, f"{make_name}{ext}")
        
        # If that file exists on your computer, load it!
        if os.path.exists(file_path):
            return Image.open(file_path)
            
    return None # Return nothing if no file is found

def generate_placeholder(make_name):
    """Creates a backup image if you haven't downloaded the logo yet"""
    img = Image.new('RGB', (600, 400), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((50, 180), f"No Image Found:\ncar_logos/{make_name}.png", fill=(100, 100, 100))
    d.rectangle([10, 10, 590, 390], outline="gray", width=4)
    return img

# --- 3. MAIN APP LAYOUT ---
st.title(" German Car Price Predictor")
st.markdown("---")

df = df.dropna(subset=['make', 'model', 'fuel', 'gear', 'offerType'])

col_left, col_right = st.columns([1, 1.5], gap="large")

# --- RIGHT SIDE: INPUTS ---
with col_right:
    st.header("Car Details")
    
    row1_1, row1_2 = st.columns(2)
    with row1_1:
        make = st.selectbox("Car Brand", sorted(df['make'].unique())) #Creates a dropdown
    with row1_2:
        models = sorted(df[df['make'] == make]['model'].unique()) #conditional filtering
        model = st.selectbox("Car Model", models)

    row2_1, row2_2 = st.columns(2)
    with row2_1:
        year = st.number_input("Year", 2000, 2024, 2019)
    with row2_2:
        mileage = st.number_input("Mileage (km)", 0, 500000, 50000)

    row3_1, row3_2 = st.columns(2)
    with row3_1:
        hp = st.number_input("Horsepower (HP)", 50, 1000, 150)
    with row3_2:
        fuel = st.selectbox("Fuel Type", sorted(df['fuel'].unique()))

    row4_1, row4_2 = st.columns(2)
    with row4_1:
        gear = st.selectbox("Transmission", sorted(df['gear'].unique()))
    with row4_2:
        offer_type = st.selectbox("Offer Type", sorted(df['offerType'].unique()))

    st.markdown("###") 
    predict_clicked = st.button(" Predict Price", type="primary", use_container_width=True)

# --- LEFT SIDE: IMAGE DISPLAY ---
with col_left:
    st.write("##") 
    
    # HERE IS THE MAGIC:
    # It calls the function to look for "car_logos/[make].png"
    car_img = get_car_image(make)
    
    if car_img:
        st.image(car_img, caption=f"{make} ", use_container_width=True)
    else:
        # If you forgot to download that specific car image, show placeholder
        placeholder = generate_placeholder(make)
        st.image(placeholder, caption=f"Missing: {make}.png", use_container_width=True)

# --- PREDICTION RESULT ---
if predict_clicked:
    input_data = pd.DataFrame([[make, model, year, mileage, hp, fuel, gear, offer_type]],
                              columns=['make', 'model', 'year', 'mileage', 'hp', 'fuel', 'gear', 'offerType'])
    
    try:
        prediction = pipe.predict(input_data)
        price = prediction[0]
        
        with col_right:
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color:#d4edda; padding:15px; border-radius:10px; border:2px solid #28a745; text-align:center;">
                <h4 style="color:#155724; margin:0;">Estimated Market Value</h4>
                <h1 style="color:#28a745; font-size:40px; margin:0;">€{price:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        with col_right:
            st.error(f"Error: {e}")