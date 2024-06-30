from fastapi import FastAPI,Request
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origin = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origin,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)
df = pd.read_csv("Customer-Churn.csv")
async def univariate_data_cou(df:pd.DataFrame,groupby:str,col:str,title:str):
    groupby_data = df.groupby(groupby)[col].count()
    label_names = groupby_data.index.tolist()
    data = groupby_data.values.tolist()
    return_data = {
        "labels":label_names,
        "data":data
        ,"label":title
    }
    return return_data
async def univariate_data_sum(df:pd.DataFrame,groupby:str,col:str,title:str):
    groupby_data = df.groupby(groupby)[col].sum()
    label_names = groupby_data.index.tolist()
    data = groupby_data.values.tolist()
    return_data = {
        "labels":label_names,
        "data":data
        ,"label":title
    }
    return return_data
async def univariate_data_avg(df:pd.DataFrame,groupby:str,col:str,title:str):
    groupby_data = df.groupby(groupby)[col].mean()
    label_names = groupby_data.index.tolist()
    data = groupby_data.values.tolist()
    
    return_data = {
        "labels":label_names,
        "data":data
        ,"label":title
    }
    return return_data
def monthCount(x):
    if(x>0 and x<=10):
        return "TenMonth"
    elif (x>10 and x<=20):
        return "TwentyMonth"
    elif (x>20 and x<=30):
        return "ThirtyMonth"
    elif(x>30 and x<=40):
        return "FourtyMonth"
    elif (x>40 and x<=50):
        return "FiftyMonth"
    elif (x>50 and x<=60):
        return "SixtyMonth"
    else:
        return "SeventyMonth"
df['tenureInTenth'] = df['tenure'].map(monthCount)
df_is_churn = df[df['Churn']=='Yes']
df=df[df["TotalCharges"]!=' ']
df["TotalCharges"]=df["TotalCharges"].astype(float)
@app.get("/")
async def get_data():
 
    onlineSecurityByChurn =await univariate_data_cou(df_is_churn,"OnlineSecurity","Churn","OnlineSecurtiyPerChurn")
    TotalChargesByGender= await univariate_data_sum(df,"gender","TotalCharges","TotalChargesByGender")
    avg_monthly_charge = await univariate_data_avg(df,"Contract","MonthlyCharges","Average MonthlyChargesPerContract")
    CustomerTenureByMonth = await univariate_data_cou(df,"tenureInTenth","customerID","CustomerTenureByMonth")
    AvgTotalChargesperTenure= await univariate_data_avg(df,"tenureInTenth","TotalCharges","AvgTotalChargesperTenure")
    GenderByChurn = await univariate_data_cou(df_is_churn,"gender","Churn","GenderByChurn ")
    CustomerChurnperTenure = await univariate_data_cou(df_is_churn,"tenureInTenth","Churn","CustomerChurnperTenure")
    return {"data":{
        "onlineSecurityByChurn":onlineSecurityByChurn,
        "TotalChargesByGender":TotalChargesByGender,
        "avg_monthly_charge":avg_monthly_charge,
        "CustomerTenureByMonth":CustomerTenureByMonth,
        "AvgTotalChargesperTenure":AvgTotalChargesperTenure,
	"GenderByChurn":GenderByChurn,
	"CustomerChurnperTenure":CustomerChurnperTenure
    }}
@app.get("/test")
async def get_data(help):
    print(f"{help}ByChurn")
    onlineSecurityByChurn =await univariate_data_cou(df_is_churn,help,"Churn",f"{help}"+" Per Churn".title())
    return {"data":{
        f"{help}ByChurn":onlineSecurityByChurn,
       
    }}
def convert_types(params: dict) -> dict:
    int_keys = [
        'gender', 'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'DeviceProtection', 'StreamingTV', 'Contract', 
        'PaperlessBilling', 'PaymentMethod'
    ]
    float_keys = ['MonthlyCharges', 'TotalCharges']

    for key in int_keys:
        if key in params:
            params[key] = int(params[key])
    
    for key in float_keys:
        if key in params:
            params[key] = float(params[key])
    
    return params
async def machineModel(data:dict):    
    df = pd.read_csv("Customer-Churn.csv")
    df.drop(columns=['customerID', 'SeniorCitizen' ,'Partner' ,'Dependents' ,'tenure' ],inplace  = True)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges']=df['TotalCharges'].astype('float')
    df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace = True)
    column_to_encode = ['PhoneService','gender'	,'MultipleLines','Contract','PaperlessBilling'	,'PaymentMethod','InternetService','OnlineSecurity','OnlineBackup',	'DeviceProtection'	,'TechSupport',	'StreamingTV'	,'StreamingMovies']
    df_encoded = df.copy()
    for col in column_to_encode:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    scaler_standard = StandardScaler()
    df_standard_scale = df_encoded.copy()
    df_standard_scale['MonthlyCharges'] = scaler_standard.fit_transform(df_standard_scale.MonthlyCharges.values.reshape(-1,1))
    df_standard_scale['TotalCharges'] = scaler_standard.fit_transform(df_standard_scale.TotalCharges.values.reshape(-1,1))
    df_standard_scale.drop(columns = ['OnlineBackup','StreamingMovies','TechSupport'],inplace=True)
    X = df_standard_scale.iloc[:, :-1].values
    y = df_standard_scale.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    svm = SVC(kernel ='rbf', random_state = 0)  


    svm.fit(X_train, y_train)
    custom_df = pd.DataFrame([data])
    custom_df_encoded = pd.get_dummies(custom_df, drop_first=True)
    df_enc = df_encoded.copy()
    df_enc.drop(columns=['OnlineBackup', 'StreamingMovies', 'TechSupport'], inplace=True)
    custom_df_encoded = pd.get_dummies(custom_df, drop_first=True)
    custom_df_encoded = custom_df_encoded.reindex(columns=df_enc.columns[:-1], fill_value=0)

    # Standardize 'MonthlyCharges' and 'TotalCharges' for custom input using the same scaler
    custom_df_encoded['MonthlyCharges'] = scaler_standard.transform(custom_df_encoded[['MonthlyCharges']])
    custom_df_encoded['TotalCharges'] = scaler_standard.transform(custom_df_encoded[['TotalCharges']])
    custom_df_encoded
    y_pred = svm.predict(custom_df_encoded.values)
    print(data)
    return y_pred[0]
@app.get("/pred")
async def get_data(pre:Request):
    temp = pre.query_params
    print("Hello")
    data = dict(temp)
    data = convert_types(data)
    print(data)
    temp =await machineModel(data)
    return {"pred":temp}