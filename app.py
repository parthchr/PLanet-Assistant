# app.py
import streamlit as st
import auth
import planetAgent_fixed  # Importing your main application

# MUST be the very first Streamlit command
st.set_page_config(page_title="Planet API Explorer", page_icon="🌍", layout="wide")

# Initialize Session State Variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'landing'
if 'username' not in st.session_state:
    st.session_state.username = ''

# --- CSS STYLING FOR A PROFESSIONAL LOOK ---
def apply_custom_css():
    st.markdown("""
        <style>
        .hero-container {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 15px;
            color: white;
            margin-bottom: 3rem;
        }
        .hero-title { font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; }
        .hero-subtitle { font-size: 1.2rem; font-weight: 300; opacity: 0.9; }
        </style>
    """, unsafe_allow_html=True)

# --- PAGE NAVIGATION FUNCTIONS ---
def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.current_page = 'landing'
    st.rerun()

# --- PAGE VIEWS ---
def show_landing_page():
    apply_custom_css()
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🌍 Planet API Explorer</div>
            <div class="hero-subtitle">Enterprise-grade satellite imagery discovery powered by natural language and Groq AI.</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Welcome to the Platform")
        st.write("Please authenticate to access the geospatial dashboard and AI analyst tools.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Log In", use_container_width=True, type="primary"):
                navigate_to('login')
        with c2:
            if st.button("Sign Up", use_container_width=True):
                navigate_to('signup')

def show_login_page():
    st.markdown("<h2 style='text-align: center;'>Sign In</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Log In", use_container_width=True)

            if submit:
                if auth.verify_login(user, pwd):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        
        if st.button("← Back to Home"):
            navigate_to('landing')

def show_signup_page():
    st.markdown("<h2 style='text-align: center;'>Create an Account</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("signup_form"):
            user = st.text_input("Username *", help="Must be unique")
            email = st.text_input("Email (optional)")
            pwd = st.text_input("Password *", type="password")
            submit = st.form_submit_button("Create Account", use_container_width=True)

            if submit:
                if len(user) < 3 or len(pwd) < 5:
                    st.warning("Username must be 3+ chars and password 5+ chars.")
                else:
                    success = auth.create_user(user, email, pwd)
                    if success:
                        st.success("Account created! Please log in.")
                        st.session_state.current_page = 'login'
                        st.rerun()
                    else:
                        st.error("Username already exists. Please choose another.")
        
        if st.button("← Back to Home"):
            navigate_to('landing')

# --- MAIN ROUTING LOGIC ---
if not st.session_state.logged_in:
    if st.session_state.current_page == 'landing':
        show_landing_page()
    elif st.session_state.current_page == 'login':
        show_login_page()
    elif st.session_state.current_page == 'signup':
        show_signup_page()
else:
    # If logged in, show the logout button in the sidebar, then run the main app!
    st.sidebar.markdown(f"**User:** {st.session_state.username}")
    if st.sidebar.button("🚪 Log Out"):
        do_logout()
    
    # Execute the main application logic
    planetAgent_fixed.main()