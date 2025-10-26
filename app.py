from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from pymongo import MongoClient
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.secret_key = "secret_key_here"

# -------------------- MongoDB Connection --------------------
client = MongoClient("mongodb+srv://adim83876_db_user:GN2TOdlN9VpNmV7M@cluster0.cu5ai2t.mongodb.net/?appName=Cluster0")

# Access your database and collection
db = client["HPS"]
users_collection = db["users"]
# -------------------- Load Dataset --------------------
df = pd.read_csv('Data.csv')  # Make sure your Data.csv file has 'State' and 'City' columns

# Clean up column names
df.columns = df.columns.str.strip()

# Extract unique states and cities
unique_states = sorted(df['State'].dropna().unique())

# -------------------- Model Training --------------------
features = ['State', 'City', 'Property_Type', 'BHK', 'Furnished_Status',
            'Size_in_SqFt', 'Public_Transport_Accessibility', 'Parking_Space', 'Security']
categorical_cols = ['State', 'City', 'Property_Type', 'BHK', 'Furnished_Status',
                    'Public_Transport_Accessibility', 'Parking_Space', 'Security']
numerical_cols = ['Size_in_SqFt']

X = df[features]
y = df['Price_in_Lakhs']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X[categorical_cols])
X_num = X[numerical_cols].values
X_processed = np.hstack([X_num, X_cat])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_processed, y)

# -------------------- Registration --------------------
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        captcha_input = request.form.get('captcha_input')
        captcha_answer = request.form.get('captcha_answer')

        if len(mobile) != 10:
            flash("Mobile number must be 10 digits", "error")
            return render_template('reg.html')
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template('reg.html')
        if captcha_input != captcha_answer:
            flash("CAPTCHA is incorrect", "error")
            return render_template('reg.html')

        if users_collection.find_one({"mobile": mobile}):
            flash("Mobile number already registered. Go for Sign In", "error")
            return render_template('reg.html')

        users_collection.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "mobile": mobile,
            "password": password
        })

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('reg.html')

# -------------------- Login --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        mobile = request.form.get("mobile")
        password = request.form.get("password")
        captcha_input = request.form.get("captcha_input")
        captcha_answer = request.form.get("captcha_answer")

        if captcha_input != captcha_answer:
            flash("Invalid CAPTCHA! Please try again.", "danger")
            return redirect(url_for("login"))

        user = users_collection.find_one({"mobile": mobile})
        if user and user["password"] == password:
            session["user"] = user["first_name"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid mobile or password!", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

# -------------------- Forgot Password --------------------
@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    mobile = request.form.get('mobile', '')
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not re.fullmatch(r'\d{10}', mobile):
        flash("Enter a valid 10-digit mobile number.", "error")
        return redirect(url_for('index'))

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for('index'))

    user = users_collection.find_one({'mobile': mobile})
    if user:
        users_collection.update_one({'mobile': mobile}, {'$set': {'password': password}})
        flash("Password reset successful!", "success")
    else:
        flash("Mobile number not registered.", "error")

    return redirect(url_for('login'))

# -------------------- City Dropdown AJAX --------------------
@app.route('/get_cities', methods=['POST'])
def get_cities():
    state = request.json.get('state')
    cities = sorted(df[df['State'] == state]['City'].dropna().unique().tolist())
    return jsonify({'cities': cities})


# -------------------- Index Page --------------------
@app.route('/index', methods=['GET', 'POST'])
def index():
    if "user" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    predicted_price = None

    if request.method == 'POST':
        state = request.form['state']
        city = request.form['city']
        property_type = request.form['propertyType']
        bhk = request.form['bhk']
        furnished = request.form['furnished']
        size = float(request.form['size'])
        transport = request.form['transport']
        parking = request.form['parking']
        security = request.form['security']

        new_data = pd.DataFrame({
            'State': [state],
            'City': [city],
            'Property_Type': [property_type],
            'BHK': [bhk],
            'Furnished_Status': [furnished],
            'Size_in_SqFt': [size],
            'Public_Transport_Accessibility': [transport],
            'Parking_Space': [parking],
            'Security': [security]
        })

        new_cat = encoder.transform(new_data[categorical_cols])
        new_num = new_data[numerical_cols].values
        new_processed = np.hstack([new_num, new_cat])

        predicted_price = model.predict(new_processed)[0]
        predicted_price = round(predicted_price, 2)  # ✅ Round to 2 decimal places

        return render_template("result.html", predicted_price=predicted_price)

    return render_template("index.html", username=session["user"], states=unique_states)


# -------------------- Logout --------------------
@app.route('/logout')
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# -------------------- Property Page --------------------

# -------------------- Property Page --------------------
@app.route('/property', methods=['GET', 'POST'])
def property():
    return render_template("property.html", states=unique_states)


@app.route('/property_result', methods=['POST'])
def property_result():
    # Load dataset
    df = pd.read_csv("Data.csv")

    # Get user inputs
    state = request.form.get('state')
    city = request.form.get('city')
    low_price = float(request.form.get('lowPrice'))
    high_price = float(request.form.get('highPrice'))
    bhk = int(request.form.get('bhk').split()[0])  # "3 BHK" → 3

    # Ensure required columns exist
    required_cols = [
        'State', 'City', 'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
        'Furnished_Status', 'Parking_Space', 'Public_Transport_Accessibility',
        'Security', 'Availability_Status'
    ]
    for col in required_cols:
        if col not in df.columns:
            return f"❌ Error: '{col}' column not found in dataset."

    # Convert data types
    df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
    df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')

    # 🏠 Apply filters
    filtered = df[
        (df['State'] == state) &
        (df['City'] == city) &
        (df['BHK'] == bhk) &
        (df['Price_in_Lakhs'] >= low_price) &
        (df['Price_in_Lakhs'] <= high_price)
    ][required_cols]  # Show only selected columns

    # Handle empty results
    if filtered.empty:
        return render_template(
            "property_result.html",
            message="No properties found matching your criteria.",
            total=0,
            avg_price=0,
            tables=None
        )

    # Stats
    total = len(filtered)
    avg_price = round(filtered['Price_in_Lakhs'].mean(), 2)

    # Render
    return render_template(
        "property_result.html",
        message=None,
        total=total,
        avg_price=avg_price,
        tables=[filtered.to_html(classes='table table-bordered table-striped text-center align-middle', index=False)]
    )



# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
