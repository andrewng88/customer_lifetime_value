
# Core Pkg
import streamlit as st
from fbprophet import Prophet

# Load EDA Pkgs
import pandas as pd 
import numpy as np

# Load Data Vis Pkg
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from fbprophet.plot import plot_plotly

import datetime as dt
import plotly.express as px
import base64

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset,encoding = "ISO-8859-1",parse_dates = ['InvoiceDate'])
    df['Revenue']=df['Quantity']*df['UnitPrice']
    df=df[df['Country']=='United Kingdom'].reset_index(drop=True)
    #convert to date only no HH:MM
    df['InvoiceDate'] = df['InvoiceDate'].dt.date
    return df

def main():
    """Customer Lifetime Value & Sales Revenue Forecasting"""

    st.title("Customer Lifetime Value & Sales Revenue Forecasting")
    st.subheader("Built with Streamlit,Lifetimes, fbProphet and Plotly library")

    # Menu
    menu = ['Exploratory Data Analysis','Customer Lifetime Value','Sales Revenue Forecasting','About']
    choices = st.sidebar.selectbox('Select Menu',menu)

    if choices == 'Exploratory Data Analysis':
        st.subheader('Exploratory Data Analysis')

        clean=pd.read_csv('data/clean_df.csv')
        clean=clean.drop('Unnamed: 0',axis=1)
        clean=clean.rename(columns={"Price":"Revenue"})
        clean["Date"]=pd.to_datetime(clean["Date"])
        clean["Month"]=clean["Date"].dt.strftime("%B")
        
        if st.checkbox('View Data'):
            st.dataframe(clean)

        st.subheader("Annual Aggregation")

        if st.checkbox('View Top 10 Items By Revenue'):
            revenue=clean.groupby("Description")["Revenue"].sum().reset_index().sort_values(by="Revenue",ascending=False)
            revenue_head=revenue.head(10).sort_values(by="Revenue")
            fig1=px.bar(revenue_head,x="Revenue",y="Description",orientation="h")
            st.plotly_chart(fig1)

        if st.checkbox('View Bottom 10 Items By Revenue'):
            revenue=clean.groupby("Description")["Revenue"].sum().reset_index().sort_values(by="Revenue",ascending=False)
            revenue_tail=revenue.tail(10).sort_values(by="Revenue")
            fig2=px.bar(revenue_tail,x="Revenue",y="Description",orientation="h")
            st.plotly_chart(fig2)

        if st.checkbox('View Top 10 Popular Items'):
            quantity=clean.groupby("Description")["Quantity"].sum().reset_index().sort_values(by="Quantity",ascending=False)
            quantity_head=quantity.head(10).sort_values(by="Quantity")
            fig3=px.bar(quantity_head,x="Quantity",y="Description",orientation="h")
            st.plotly_chart(fig3)

        if st.checkbox('View Least Popular Items'):
            qty1=st.selectbox("Select Total Quantity Sold",[1,2,3,4,5,6,7,8,9,10],key="qty1")
            quantity=clean.groupby("Description")["Quantity"].sum().reset_index().sort_values(by="Quantity",ascending=False)
            quantity_tail=quantity[quantity["Quantity"]==qty1].reset_index(drop=True)
            st.dataframe(quantity_tail[["Description"]])

        st.subheader("Monthly Aggregation")

        if st.checkbox('View Monthly Top 10 Items By Revenue'):
            mth1=st.selectbox("Select Month",["January","February","March","April","May","June","July","August","September","October","November","December"],key="mth1")
            monthrevenue=clean.groupby(["Month","Description"])["Revenue"].sum().reset_index()
            month_revenue=monthrevenue[monthrevenue["Month"]==mth1].sort_values(by="Revenue",ascending=False)
            month_revenue_head=month_revenue.head(10).sort_values(by="Revenue") 
            fig4=px.bar(month_revenue_head,x="Revenue",y="Description",orientation="h")
            st.plotly_chart(fig4)

        if st.checkbox('View Monthly Bottom 10 Items by Revenue'):
            mth2=st.selectbox("Select Month",["January","February","March","April","May","June","July","August","September","October","November","December"],key="mth2")
            monthrevenue=clean.groupby(["Month","Description"])["Revenue"].sum().reset_index()
            month_revenue=monthrevenue[monthrevenue["Month"]==mth2].sort_values(by="Revenue",ascending=False)
            month_revenue_tail=month_revenue.tail(10).sort_values(by="Revenue")   
            fig5=px.bar(month_revenue_tail,x="Revenue",y="Description",orientation="h")
            st.plotly_chart(fig5)

        if st.checkbox('View Monthly Top 10 Popular Items'):
            mth3=st.selectbox("Select Month",["January","February","March","April","May","June","July","August","September","October","November","December"],key="mth3")
            monthquantity=clean.groupby(["Month","Description"])["Quantity"].sum().reset_index()
            month_quantity=monthquantity[monthquantity["Month"]==mth3].sort_values(by="Quantity",ascending=False)
            month_quantity_head=month_quantity.head(10).sort_values(by="Quantity")      
            fig6=px.bar(month_quantity_head,x="Quantity",y="Description",orientation="h")
            st.plotly_chart(fig6)
        
        if st.checkbox('View Monthly Least Popular Items'):
            mth4=st.selectbox("Select Month",["January","February","March","April","May","June","July","August","September","October","November","December"],key="mth4")
            qty2=st.selectbox("Select Total Quantity Sold",[1,2,3,4,5,6,7,8,9,10],key="qty2")
            monthquantity=clean.groupby(["Month","Description"])["Quantity"].sum().reset_index()
            month_quantity_tail=monthquantity[(monthquantity["Month"]==mth4)&(monthquantity["Quantity"]==qty2)].reset_index(drop=True)   
            st.dataframe(month_quantity_tail[["Description"]])

    if choices == 'Customer Lifetime Value':
        st.subheader('Customer Lifetime Value')
        st.subheader("Model Based On 30 Days")

        output=pd.read_csv('data/output_df.csv')
        output["predicted_purchases"]=output["predicted_purchases"].round()
        output["expected_total_monetary_value"]=output["predicted_purchases"]*output["expected_monetary_value"]
        #output=output.rename(columns={"probability":"probability_alive"})

        if st.checkbox('View Predictions'):
            #st.dataframe(output[["CustomerID","predicted_purchases","expected_monetary_value","expected_total_monetary_value","probability_alive"]])
            st.dataframe(output[["CustomerID","predicted_purchases","expected_monetary_value","expected_total_monetary_value"]])
            def get_table_download_link(df):
                csv=df.to_csv(index=False)
                b64=base64.b64encode(csv.encode()).decode()  
                return f'<a href="data:file/csv;base64,{b64}" download="data/output_df.csv">Download</a>'
            st.markdown(get_table_download_link(output),unsafe_allow_html=True)

        if st.checkbox('View More On Expected Total Monetary Value'):
            exp_tot=output["expected_total_monetary_value"].describe().to_frame()
            st.dataframe(exp_tot)

            st.subheader("Boxplot")
            fig7=px.box(output,y="expected_total_monetary_value")
            st.plotly_chart(fig7)

            st.subheader("Histogram")
            fig8=px.histogram(output,x="expected_total_monetary_value")
            st.plotly_chart(fig8)

    if choices == 'Sales Revenue Forecasting':
        st.subheader('Sales Revenue Forecasting')

        df_load_state = st.text('Loading data...')
        df = load_data('data/data.csv')
        df_load_state.text('Loading data... done!') 
        
        chart = df.groupby(['InvoiceDate'])[['Revenue']].sum()
        
        def plot_fig():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart.index, y=chart['Revenue'], name="Revenue"))
            fig.layout.update(title_text='UK Revenue for year 2011 ',xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            return fig
        
        # plotting the figure of Actual Data
        plot_fig()
        
        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(chart)

        #shape the df w.r.t requirement by fbProphet
        df_prophet = df.groupby(['InvoiceDate'],as_index=False)[['Revenue']].sum()

        #remove negative value
        #fbprophet works with 'None'
        df_prophet.iloc[21,1]=None
        df_prophet.columns = ['ds','y']
        
        #function to remove outliers
        def outliers_to_na(ts, devs):
            median= ts['y'].median()
            #print(median)
            std = np.std(ts['y'])
            #print(std)
            for x in range(len(ts)):
                val = ts['y'][x]
                #print(ts['y'][x])
                if (val < median - devs * std or val > median + devs * std):
                    ts['y'][x] = None 
            return ts

        # remove outliers based on 2 std dev
        outliers_to_na(df_prophet , 2)

        #st.write(df_prophet)

        #season_choice = st.selectbox('Seasonality Mode',['additive','multiplicative'])
        #model_choice = st.selectbox('Model Choice',['Logistic Regression','Neural Network'])

        #if changepoint_prior_scale == 'additive':
        m = Prophet(seasonality_mode='additive',changepoint_prior_scale=0.11)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=3,freq='M')
        future = m.predict(future)

        #plot forecast
        fig1 = plot_plotly(m, future)
        if st.checkbox('Show forecast data'):
            st.subheader('forecast data')
            st.write(future.loc[305:,['ds','yhat']])
            st.write('Quarterly Sales Revenue for Dec 2011, Jan 2012 , Feb 2012')
            st.plotly_chart(fig1)

        #plot component wise forecast
        st.write("Component wise forecast")
        fig2 = m.plot_components(future)
        st.write(fig2)
        
    if choices == 'About':
        st.subheader('About')

if __name__=='__main__':
    main()
