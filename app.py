import gradio as gr
import joblib
import pandas as pd

model = joblib.load('best_rf_model.pkl')

def predict(distance, prep_time, weather, traffic, vehicle,
            city, festival, time_of_day, age, rating,
            vehicle_condition, multi_delivery):

    weather_map  = {'Sunny':0,'Cloudy':1,'Windy':2,'Fog':3,'Sandstorms':4,'Stormy':5}
    traffic_map  = {'Low':0,'Medium':1,'High':2,'Jam':3}
    vehicle_map  = {'bicycle':0,'scooter':1,'motorcycle':2,'electric_scooter':3}
    city_map     = {'Semi-Urban':0,'Urban':1,'Metropolitian':2}
    tod_map      = {'Morning':0,'Afternoon':1,'Evening':2,'Night':3}
    festival_map = {'No':0,'Yes':1}

    # safe defaults
    festival       = festival or 'No'
    multi_delivery = multi_delivery or 'No'
    weather        = weather or 'Sunny'
    traffic        = traffic or 'Low'
    vehicle        = vehicle or 'motorcycle'
    city           = city or 'Urban'
    time_of_day    = time_of_day or 'Morning'

    hour       = {'Morning':10,'Afternoon':14,'Evening':19,'Night':22}[time_of_day]
    is_rush    = 1 if (11 <= hour <= 14) or (18 <= hour <= 21) else 0
    is_weekend = 0
    is_multi   = 1 if multi_delivery == 'Yes' else 0

    # exactly 16 features in exact order the model was trained with
    input_df = pd.DataFrame([[
        distance,
        prep_time,
        hour,
        is_rush,
        is_weekend,
        age,
        rating,
        is_multi,
        vehicle_condition,
        weather_map[weather],
        traffic_map[traffic],
        vehicle_map[vehicle],
        city_map[city],
        tod_map[time_of_day],
        2,
        festival_map[festival],
    ]], columns=[
        'Distance_km',
        'Prep_Time_min',
        'Hour_of_Day',
        'Is_Rush_Hour',
        'Is_Weekend',
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Is_Multiple_Delivery',
        'Vehicle_condition',
        'Weather_Code',
        'Traffic_Code',
        'Vehicle_Code',
        'City_Code',
        'Time_Day_Code',
        'Order_Type_Code',
        'Is_Festival',
    ])
    print(input_df.to_string())
    prediction = model.predict(input_df)[0]
    return f"Estimated Delivery Time: {round(prediction)} minutes"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 30, value=5,              label="Distance (km)"),
        gr.Slider(1, 60, value=10,             label="Prep Time (min)"),
        gr.Dropdown(['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy'], value='Sunny',     label="Weather"),
        gr.Dropdown(['Low','Medium','High','Jam'],                          value='Low',        label="Traffic"),
        gr.Dropdown(['bicycle','scooter','motorcycle','electric_scooter'],  value='motorcycle', label="Vehicle Type"),
        gr.Dropdown(['Semi-Urban','Urban','Metropolitian'],                 value='Urban',      label="City Type"),
        gr.Radio(['No','Yes'],                                             value='No',         label="Festival Day?"),
        gr.Dropdown(['Morning','Afternoon','Evening','Night'],              value='Morning',    label="Time of Day"),
        gr.Slider(20, 40, value=28,                label="Driver Age"),
        gr.Slider(4.0, 5.0, value=4.5, step=0.1,  label="Driver Rating"),
        gr.Slider(0, 2, value=1,                   label="Vehicle Condition (0=Poor, 2=Good)"),
        gr.Radio(['No','Yes'],                     value='No',            label="Multiple Deliveries?"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Food Delivery Time Predictor",
    description="Enter order details to get an estimated delivery time",
)

demo.launch(inbrowser=True)