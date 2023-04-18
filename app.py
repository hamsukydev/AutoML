# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import pandas_profiling
# #import profilling capability
# from streamlit_pandas_profiling import st_profile_report
# import os
#
# with st.sidebar:
#     st.image("./inno2.png")
#     st.title("AutoStreamML")
#     choice= st.radio("Navigation", ["Upload", "Profilling", "ML", "Download"])
#     st.info("This application allows you to build an automated ML pipline using Streamlit, Pandas Profilling and PyCaret. ")
#
# st.write("Hello Word")
#
# if os.path.exists("Sourcedata.csv"):
#     df= pd.read_csv("Sourcedata.csv", index_col=None)
#
# if choice == "Upload":
#     st.title("Upload Your Data for Modelling")
#     file=st.file_uploader("Upload Your Dataset Here")
#     if file:
#          df= pd.read_csv(file, index_col=None)
#          df.to_csv("Sourcedata.csv", index=None)
#          st.dataframe(df)
#
#         #do something
#
#
# if choice == "Profilling":
#     st.title("Automated Exploratory Data Analysis")
#     profile_report= df.profile_report()
#     st_profile_report(profile_report)
#
# if choice == "ML":
#     pass
#
# if choice == "Download":
#     pass


import pandas as pd
import numpy as np

x = pd.DataFrame(range(3, 10))
print(x)

n = {
        'a': 1, 's': 5,
        'd': 9,'b': 4
    }, {
        's': 5, 'd': 9,'a':
        1,
        'b': 9}

b = pd.DataFrame(n)
print(b)

