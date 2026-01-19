import os
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# 1. SETUP FOLDER
FOLDER_NAME = "car_logos"
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
    print(f"Created folder: {FOLDER_NAME}")

# 2. LIST OF ALL MAKES (From your dataset)
ALL_MAKES = [
    'BMW', 'Volkswagen', 'SEAT', 'Renault', 'Peugeot', 'Toyota', 'Opel', 'Mazda', 'Ford', 
    'Mercedes-Benz', 'Chevrolet', 'Audi', 'Fiat', 'Kia', 'Dacia', 'MINI', 'Hyundai', 'Skoda', 
    'Citroen', 'Infiniti', 'Suzuki', 'SsangYong', 'smart', 'Cupra', 'Volvo', 'Jaguar', 'Porsche', 
    'Nissan', 'Honda', 'Mitsubishi', 'Lexus', 'Jeep', 'Maserati', 'Bentley', 'Land', 'Alfa', 
    'Subaru', 'Dodge', 'Microcar', 'Lamborghini', 'Lada', 'Tesla', 'Chrysler', 'McLaren', 'Aston', 
    'Rolls-Royce', 'Lancia', 'Abarth', 'DS', 'Daihatsu', 'Ligier', 'Ferrari', 'Aixam', 'Zhidou', 
    'Morgan', 'Maybach', 'RAM', 'Alpina', 'Polestar', 'Brilliance', 'Piaggio', 'FISKER', 'Others', 
    'Cadillac', 'Iveco', 'Isuzu', 'Corvette', 'Baic', 'DFSK', 'Estrima', 'Alpine'
]

# 3. URL DICTIONARY (Real logos)
LOGO_URLS = {
    'Volkswagen': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Volkswagen_logo_2019.svg/600px-Volkswagen_logo_2019.svg.png',
    'BMW': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/600px-BMW.svg.png',
    'Mercedes-Benz': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Mercedes-Benz_Logo_2010.svg/600px-Mercedes-Benz_Logo_2010.svg.png',
    'Audi': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Audi-Logo_2016.svg/600px-Audi-Logo_2016.svg.png',
    'Ford': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Ford_logo_flat.svg/600px-Ford_logo_flat.svg.png',
    'Opel': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Opel_Logo_2011.svg/600px-Opel_Logo_2011.svg.png',
    'Renault': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Renault_Logo_2015.svg/600px-Renault_Logo_2015.svg.png',
    'Peugeot': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Peugeot_Logo_2010.svg/600px-Peugeot_Logo_2010.svg.png',
    'Toyota': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Toyota_EU.svg/600px-Toyota_EU.svg.png',
    'Fiat': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Fiat_Logo_2006.svg/600px-Fiat_Logo_2006.svg.png',
    'SEAT': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Seat_Logo_2012.svg/600px-Seat_Logo_2012.svg.png',
    'Skoda': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Skoda_Auto_Logo_2016.svg/600px-Skoda_Auto_Logo_2016.svg.png',
    'Hyundai': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Hyundai_Motor_Company_logo.svg/600px-Hyundai_Motor_Company_logo.svg.png',
    'Kia': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Kia_Motors.svg/600px-Kia_Motors.svg.png',
    'Volvo': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Volvo_Trucks_Logo.svg/600px-Volvo_Trucks_Logo.svg.png',
    'Mazda': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Mazda_Logo_1997.svg/600px-Mazda_Logo_1997.svg.png',
    'Citroen': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Citroën_2009.svg/600px-Citroën_2009.svg.png',
    'Nissan': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Nissan_logo.png/600px-Nissan_logo.png',
    'Dacia': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Dacia_2021_logo.svg/600px-Dacia_2021_logo.svg.png',
    'Land': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Land_Rover_logo_black.svg/600px-Land_Rover_logo_black.svg.png',
    'Ferrari': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Ferrari-Logo.svg/600px-Ferrari-Logo.svg.png',
    'Lamborghini': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Lamborghini_Logo.jpg/600px-Lamborghini_Logo.jpg',
    'Porsche': 'https://upload.wikimedia.org/wikipedia/de/thumb/2/2d/Porsche_Logo.svg/449px-Porsche_Logo.svg.png',
    'Tesla': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/600px-Tesla_Motors.svg.png',
    'Chevrolet': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Chevrolet-logo.png/600px-Chevrolet-logo.png',
    'Jeep': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Jeep_wordmark.svg/600px-Jeep_wordmark.svg.png',
    'Dodge': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Dodge_logo_2010.svg/600px-Dodge_logo_2010.svg.png',
    'Chrysler': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Chrysler_logo_2010.svg/600px-Chrysler_logo_2010.svg.png',
    'Jaguar': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Jaguar_Cars_logo_2012.svg/600px-Jaguar_Cars_logo_2012.svg.png',
    'Bentley': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Bentley_logo.svg/600px-Bentley_logo.svg.png',
    'Aston': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Aston_Martin_logo.svg/600px-Aston_Martin_logo.svg.png',
    'Maserati': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Maserati_logo.svg/600px-Maserati_logo.svg.png',
    'Suzuki': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Suzuki_logo.svg/600px-Suzuki_logo.svg.png',
    'Mitsubishi': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Mitsubishi_logo.svg/600px-Mitsubishi_logo.svg.png',
    'Subaru': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Subaru_logo.svg/600px-Subaru_logo.svg.png',
    'Lexus': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Lexus_division_emblem.svg/600px-Lexus_division_emblem.svg.png',
    'Alfa': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Alfa_Romeo_Brand.svg/600px-Alfa_Romeo_Brand.svg.png',
    'Lancia': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Lancia_Logo.svg/600px-Lancia_Logo.svg.png',
    'MINI': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/MINI_logo.svg/600px-MINI_logo.svg.png',
    'smart': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Smart_logo.svg/600px-Smart_logo.svg.png',
    'DS': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/DS_Automobiles_logo.svg/600px-DS_Automobiles_logo.svg.png',
    'Alpine': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Alpine_logo.svg/600px-Alpine_logo.svg.png',
    'Cupra': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Cupra_black.png/600px-Cupra_black.png',
    'Abarth': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Abarth_logo.svg/600px-Abarth_logo.svg.png',
    'Infiniti': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Infiniti_logo.svg/600px-Infiniti_logo.svg.png',
    'Isuzu': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Isuzu_Motors_Logo.svg/600px-Isuzu_Motors_Logo.svg.png',
    'Cadillac': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Cadillac_logo.svg/600px-Cadillac_logo.svg.png',
    'Rolls-Royce': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Rolls-Royce_Motor_Cars_logo.svg/600px-Rolls-Royce_Motor_Cars_logo.svg.png',
    'McLaren': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/McLaren_Automotive_logo.svg/600px-McLaren_Automotive_logo.svg.png',
    'Maybach': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Maybach_Logo.svg/600px-Maybach_Logo.svg.png',
    'RAM': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Ram_Trucks_logo.svg/600px-Ram_Trucks_logo.svg.png',
    'Corvette': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Corvette_Racing_logo.svg/600px-Corvette_Racing_logo.svg.png',
    'Iveco': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Iveco_logo.svg/600px-Iveco_logo.svg.png',
    'Morgan': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Morgan_Motor_Company_Logo.svg/600px-Morgan_Motor_Company_Logo.svg.png',
    'Daihatsu': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Daihatsu_Logo.svg/600px-Daihatsu_Logo.svg.png',
    'Polestar': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Polestar_logo.svg/600px-Polestar_logo.svg.png',
    'Alpina': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Alpina_Logo.svg/600px-Alpina_Logo.svg.png'
}

# 4. FUNCTION TO GENERATE PLACEHOLDERS
def create_placeholder_image(make_name):
    """Creates a simple image with the text name"""
    img = Image.new('RGB', (600, 400), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    # Just use default font, no external file needed
    d.text((50, 180), make_name, fill=(50, 50, 50))
    d.rectangle([10, 10, 590, 390], outline="gray", width=4)
    return img

# 5. MAIN LOOP
print("Starting download process...")

for make in ALL_MAKES:
    filename = f"{FOLDER_NAME}/{make}.png"
    
    # A. Check if we have a real URL for this car
    if make in LOGO_URLS:
        print(f"Downloading: {make}...")
        try:
            # Fake a browser user-agent to avoid being blocked by Wikipedia
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(LOGO_URLS[make], headers=headers)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {make}, generating placeholder.")
                img = create_placeholder_image(make)
                img.save(filename)
                
        except Exception as e:
            print(f"Error downloading {make}: {e}")
            img = create_placeholder_image(make)
            img.save(filename)
            
    # B. If no URL, generate a placeholder immediately
    else:
        print(f"Generating placeholder for: {make}")
        img = create_placeholder_image(make)
        img.save(filename)

print("------------------------------------------------")
print("Done! All images are saved in the 'car_logos' folder.")
print("Now run 'streamlit run dashboard.py' and it will see them!")