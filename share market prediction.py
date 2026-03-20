import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
   
df=pd.read_csv("apple_stock.csv")
print(df)
df['Date']=pd.to_datetime(df['Date'])
df=df.sort_values('Date')


x=df[['Open','High','Low','Volume']]
y=df['Close']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)
print(predictions)

error=mean_squared_error(y_test,predictions)
print("Error:",error)

