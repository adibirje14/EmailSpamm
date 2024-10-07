import streamlit as st
import pickle
import numpy as np
import os

# Load the spam detection model and vectorizer
model_path = r"https://raw.githubusercontent.com/adibirje14/EmailSpamm/master/merged_model.pkl"
vectorizer_path = r"vectorizer.pkl"

# Ensure the model and vectorizer files are available
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

model = load(model_path)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict spam
def predict_spam(email_content):
    # Transform the email content using the loaded vectorizer
    input_data_vectorized = vectorizer.transform([email_content])
    
    # Predict using the loaded model
    prediction = model.predict(input_data_vectorized)[0]
    probability = model.predict_proba(input_data_vectorized)[0]
    
    return prediction, probability

st.set_page_config(layout="wide")

# Load CSS at the beginning to minimize flicker
css_path = 'home.css'
if os.path.exists(css_path):
    with open(css_path) as c:
        st.markdown(f"<style>{c.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found!")

# HTML for Header and UI layout
html_content = """
<header>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Tajawal:wght@300&display=swap" rel="stylesheet">
    <div class="container">
        <div class="logo">
            <h2>Email Spam Detection System</h2>
        </div>
        <nav class="nav-a">
            <ul>
                <li><a href="home.html">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </div>
</header>
<section class="slider" id="slider">
    <div class="gif-container">
        <div class="gif-item" style="background-image: url('https://media.kasperskydaily.com/wp-content/uploads/sites/92/2013/10/06015546/21.gif');"></div>
        <div class="gif-item" style="background-image: url('https://www.rd.com/wp-content/uploads/2020/12/SMS-pop-new.gif?fit=300,300&webp=1');"></div>
    </div>
    <div class="slider-text">Accurately identify spam emails using Machine Learning Algorithms like Random Forest, Naive Bayes, Support Vector Machine, Logistic Regression & XGBoost.</div>
</section>
<section class="about" id="about">
    <div><h2>About Us</h2></div>
    <div class="sentence">This spam detection tool helps you identify potential spam emails by thoroughly analyzing the content, using advanced algorithms to ensure accurate classification and inbox security.</div>
</section>
<div class="container1">
    <section class="services">
        <div class="service-text">
            <h1>Our Services</h1>
            <p>Our system accurately detects and filters spam emails using machine learning algorithms, ensuring a cleaner inbox and enhanced email security by identifying unwanted or malicious messages.</p>
            <h2>Email Spam Detection</h2>
            <h3>Created by Gaurang - A701, Soham - A705, Aditya - A713, Sumit - A715</h3>
        </div>
    </section>    
</div>
"""

st.markdown(html_content, unsafe_allow_html=True)

# Use session state to keep track of the email body
if "email_body" not in st.session_state:
    st.session_state.email_body = ""

# Single text area for the email body
email_body = st.text_area("Enter the Email Content", st.session_state.email_body)

# Button to trigger spam detection
class_btn = st.button("Detect Spam")

if class_btn:
    st.session_state.email_body = email_body  # Save email body in session state
    if not email_body:
        st.write("Please provide the email content.")
    else:
        with st.spinner('Analyzing...'):
            try:
                prediction, probability = predict_spam(email_body)
                if prediction == 1:
                    st.success(f"Spam detected with probability: {np.round(probability[1] * 100, 2)}%")
                else:
                    st.success(f"Not spam with probability: {np.round(probability[0] * 100, 2)}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer and Contact Section
html_content1 = """
<section id="contact">
    <div class="container-contact">
        <div class="contact-message">
            <h2>Contact Us</h2>
            <p>For any inquiries or feedback, feel free to reach out to us.</p>
        </div>
        <div class="contact-form">
            <form action="#" method="POST">
                <div class="form-group">
                    <div class="half1">
                        <input type="text" name="first_name" placeholder="First Name" required>
                    </div>
                    <div class="half2">
                        <input type="text" name="last_name" placeholder="Last Name" required>
                    </div>
                </div>
                <div class="form-group">
                    <input type="email" name="email" placeholder="Your Email" required>
                </div>
                <div class="form-group">
                    <textarea name="message" placeholder="Your Message" rows="5" required></textarea>
                </div>
                <div class="form-group">
                    <button type="submit">Submit</button>
                </div>
            </form>
        </div>
    </div>
</section>
<footer>
    <div class="footer">
        <div class="footer-text">
            <p>Copyright &copy; 2024 Email Spam Detection System| All Rights Reserved.</p>
        </div>
    </div>
</footer>
"""

st.markdown(html_content1, unsafe_allow_html=True)

# Load JS if available
js_path = 'home.js'
if os.path.exists(js_path):
    with open(js_path) as j:
        st.markdown(f"<script>{j.read()}</script>", unsafe_allow_html=True)
else:
    st.warning("JS file not found!")
