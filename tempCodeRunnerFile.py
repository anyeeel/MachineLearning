from flask import Flask, render_template, url_for, redirect, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
import numpy as np
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import pickle
import base64
from sklearn import preprocessing


from models.ml_model import clean_text

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

try:
    knn_model = joblib.load('classification_model.joblib')
    scaler = joblib.load('C:/Users/Anjel69/Desktop/machinelearning/scaler.pkl')
    linreg = joblib.load('C:/Users/Anjel69/Desktop/machinelearning/linreg.pkl')
except FileNotFoundError:
    print("Model or scaler file not found.")
    knn_model = None
    scaler = None
    linreg = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    knn_model = None
    scaler = None
    linreg = None

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the model
knn_model = joblib.load('classification_model.joblib')

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=50)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    login_success = None

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            login_success = 'success'
            return redirect(url_for('dashboard'))
        else:
            login_success = 'failure'

    return render_template('login.html', form=form, login_success=login_success)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    symptoms = ""
    prediction = ""  # Define prediction outside of the if statement
    wordcloud_img = None

    if request.method == 'POST':
        symptoms = request.form['symptoms']
        disease, symptoms_for_disease = make_prediction(knn_model, symptoms)
        wordcloud_img = generate_wordcloud(symptoms_for_disease)  # Pass symptoms for word cloud generation
          # Update the prediction with the predicted disease
        prediction = disease

    return render_template('dashboard.html', symptoms=symptoms, prediction=prediction, wordcloud_img=wordcloud_img)

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load your dataset
df = pd.read_csv('Symptom2Disease.csv')

def make_prediction(model, text):
    text = clean_text(text)
    tfidf = tfidf_vectorizer.transform([text]).toarray()
    disease = model.predict(tfidf)[0]
    
    # Fetch symptoms for the predicted disease from the dataset
    symptoms_for_disease = df[df['label'] == disease]['text'].values.tolist()

    return disease, symptoms_for_disease


def generate_wordcloud(symptoms):
    # Check if symptoms have some text data
    if symptoms:
        # Combine all symptoms into a single text
        text_data = " ".join(symptoms)

        # Create Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

        # Convert the Word Cloud to an image
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_str = base64.b64encode(img.read()).decode('utf-8')

        return f"data:image/png;base64,{img_str}"
    else:
        # Return a default word cloud image or handle as needed
        return None



from sklearn.preprocessing import StandardScaler
#Train Test Split
from sklearn.model_selection import train_test_split

scaler = joblib.load('C:/Users/Anjel69/Desktop/machinelearning/scaler.pkl')
linreg = joblib.load('C:/Users/Anjel69/Desktop/machinelearning/linreg.pkl')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)




@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        if scaler is None or linreg is None:
            print("Model or scaler is not loaded.")
            return render_template('regression.html', prediction="Error: Model or scaler not loaded.")

        # Get user input from the form
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        DC = float(request.form['DC'])
        ISI = float(request.form['ISI'])
        Region = float(request.form['Region'])

        # Scale the user input
        scaled_input = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, Region]])

        # Make predictions
        prediction = linreg.predict(scaled_input)

        # Print or use the prediction as needed
        print("Predicted Forest Fire: ", prediction[0])

        # You can pass the prediction to your template or render a new template with the result
        return render_template('regression.html', prediction=prediction[0])

    # If it's a GET request, just render the regression form
    return render_template('regression.html')




    '''
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")

'''
    '''
    temperature = None
    rh = None
    rain = None
    prediction = None

    if request.method == 'POST':
        # Get input values from the form
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        rain = float(request.form['rain'])

        # Use the scaler to preprocess the input features
        input_features = scaler.transform([[temperature, rh, rain]])

        # Make predictions using the machine learning model
        prediction = forest_fire_model.predict(input_features)[0]

    return render_template("regression.html", temperature=temperature, rh=rh, rain=rain, prediction=prediction)
   
    '''


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

from flask import flash

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # Add SweetAlert for success
        flash('Registration successful', 'success')  # Use flash to display a message after redirection
        return redirect(url_for('login'))
    else:
        # Add SweetAlert for failure
        flash('Registration failed. Username already exists. Please choose a different one.', 'error')

    return render_template('register.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
