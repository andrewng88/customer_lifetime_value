FROM python:3.7

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt update && apt -y install build-essential 

RUN pip install -r requirements.txt

RUN pip install convertdate

RUN pip install LunarCalendar

RUN pip install holidays

RUN pip install pystan==2.19 

RUN pip install fbprophet==0.6

COPY . .

CMD streamlit run app.py