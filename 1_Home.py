"""

Content- articles
computation 
interface


That means:

reasoning logic (core/)

interface experimentation (pages/)

reusable UI components (components/)

experimentation sandbox (experiments/)

If you don't separate these now, you’ll regret it in 3 months.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "sample.txt"

"""



import streamlit as st

st.set_page_config(
    page_title="ALGORITHM INSTITUTE",
    layout="wide"
)

st.title("Welcome to the Algorithm Institute")

# st.markdown("""
#     <style>
#         [data-testid="stSidebarNav"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

st.markdown("""
This site contains:

- 📘 Articles
- 🎓 Lessons
- ⚙️ Interactive simulations
- 🤖 Stock Market Analytics
""")

# st.header("Algorithms")

# st.page_link("pages/2_algorithms.py", label="Algorithms for AI")

# st.header("Interactive Tools")
# st.page_link("pages/3_Stock_analytics.py", label="Stock Analytics")
#st.page_link("pages/4_Model_Playground.py", label="Reasoning Model Playground")
#st.page_link("pages/4_Model_Playground.py", label="Reasoning Model Playground")

#with open("content/cosmology_intro.md") as f:
#    st.markdown(f.read())

# page = st.sidebar.selectbox(
#     "Choose page",
#     ["Home", "Lesson 1", "Simulation"]
# )

# if page == "Home":
#     show_home()
# elif page == "Lesson 1":
#     show_lesson()
# elif page == "Simulation":
#     show_simulation()