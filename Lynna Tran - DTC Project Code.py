'''Lynna Tran   
OSMBA 5067 - Spring 2021
Data Translation Challenge - Project Code

'''


##load libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#Import model libraries 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB

##prevent future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##Import dataset 
data17 = pd.read_csv (r'/Users/lynnatran/Documents/MSBA/5067 - Machine Learning/Project/data/2017-04_data.csv')
data18 = pd.read_csv (r'/Users/lynnatran/Documents/MSBA/5067 - Machine Learning/Project/data/2018-04_data.csv')
print(data17, data18)




##Select variables

#VARIABLES 
'''
Primary Key: 
GEO_ID    = id

Features:

INDLEVEL           = Industry level -- no longer needed since its only one unique

SEX                = Gender of owners (INT)
SEX_LABEL          = Meaning of Sex code (Cateogorial Values )
                


RACE_GROUP         = Race code (INT)
RACE_GROUP_LABEL   = Meaning of Race code (Categorical)


EMPSZFI             = Employment size of firms code (Int)
EMPSZFI_LABEL       = Meaning of Employment size of firms code (Categorical)

QDESC              = Question description code (Who owns the business)
QDESC_LABEL        = Meaning of Question description code

##not using since there's no many values 
##BUSCHAR            = Business characteristic code
##BUSCHAR_LABEL      = Meaning of Business characteristic code

FIRMPDEMP          = Number of employer firms
FIRMPDEMP_PCT      = Percent of employer firms (%)

EMP                = Number of employees
EMP_PCT            = Percent of employees (%)


PAYANN             = Annual payroll ($1,000)
PAYANN_PCT         = Percent of annual payroll (%)

YEAR               = YEAR

'''

df17 = data17[["INDLEVEL", "SEX", "SEX_LABEL", "RACE_GROUP", "RACE_GROUP_LABEL", "EMPSZFI",
            "EMPSZFI_LABEL", "QDESC", "QDESC_LABEL", "BUSCHAR", "BUSCHAR_LABEL",
            "FIRMPDEMP", "FIRMPDEMP_PCT", "EMP", "EMP_PCT", "PAYANN", "PAYANN_PCT", "YEAR"]]

df18 = data18[["INDLEVEL", "SEX", "SEX_LABEL", "RACE_GROUP", "RACE_GROUP_LABEL", "EMPSZFI",
            "EMPSZFI_LABEL", "QDESC", "QDESC_LABEL", "BUSCHAR", "BUSCHAR_LABEL",
            "FIRMPDEMP", "FIRMPDEMP_PCT", "EMP", "EMP_PCT", "PAYANN", "PAYANN_PCT", "YEAR"]]             





##Preprocessing Variables 

##build dataset


dataset = df17.append(df18)

data_simp_cols = dataset[["SEX", 'SEX_LABEL', "RACE_GROUP", "RACE_GROUP_LABEL",
                          'EMPSZFI'  , "EMPSZFI_LABEL",
                            "QDESC",  'QDESC_LABEL', 
                            "FIRMPDEMP", "EMP", "PAYANN",  "YEAR"]] 


#list for each column feature 

Sex_list  = []
Race_list  = []
EmpSiz_list  = []
EmpSiz_Laballist = []
Owner_list  = []
BChar_list  = []
Firm_list  = []
Employee_list  = []
Pay_list = []
Year  = []

Target = []

##SEX Binary Variables List 
SEX_Total = []
SEX_Female = []
SEX_Male = []
SEX_EMF =[]
SEX_Classifiable = []
SEX_Unclassifiable = []

##Race Binary Variables 
Race_Total = []
Race_White = []
Race_Black = []
Race_Native = []
Race_Asian = []
Race_Islander = []
Race_Minority = []
Race_EMN = []
Race_Nonminority = []
Race_Classifiable = []
Race_Unclassifiable = []


##Employee Size Variables 
Firms_No     = []
Firms_1_4     = []
Firms_5_9     = []
Firms_10_19   = []
Firms_20_49   = []
Firms_50_99   = []
Firms_100_249 = []
Firms_250_499 = []
Firms_500     = []

##BUSCHAR binary variables list 
QDESC_OWNRNUM = []
QDESC_FAMOWN = []
QDESC_SPOUSES = []
QDESC_CUST = []
QDESC_WORKERS = []
QDESC_CEASEOPS = []

##Employee Size Variables list 
EMP_0_19 = []
EMP_99 = []
EMP_249 = []
EMP_499 = []
EMP_999 = []
EMP_2499 = []
EMP_4999 = []
EMP_9999 = []
EMP_24k = []
EMP_50k = []
EMP_99k = []
EMP_100k = []


missing_values = ['X', 'S', 'D']



for index, row in data_simp_cols.iterrows():
    if index != 0:
        ##Sex grouping of companies 
        Sex_list.append(row['SEX']) ##categorical value for NB
        if row['SEX_LABEL'] =='Total':    ##changing to binary value for NM
            SEX_Total.append(1)
            SEX_Female.append(0)
            SEX_Male.append(0)
            SEX_EMF.append(0)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(0)
        elif row['SEX_LABEL'] =='Female':
            SEX_Total.append(0)
            SEX_Female.append(1)
            SEX_Male.append(0)
            SEX_EMF.append(0)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(0)
        elif row['SEX_LABEL'] =='Male':
            SEX_Total.append(0)
            SEX_Female.append(0)
            SEX_Male.append(1)
            SEX_EMF.append(0)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(0)
        elif row['SEX_LABEL'] =='Equally male/female':
            SEX_Total.append(0)
            SEX_Female.append(0)
            SEX_Male.append(0)
            SEX_EMF.append(1)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(0)
        elif row['SEX_LABEL'] =='Classifiable':
            SEX_Total.append(0)
            SEX_Female.append(0)
            SEX_Male.append(0)
            SEX_EMF.append(0)
            SEX_Classifiable.append(1)
            SEX_Unclassifiable.append(0)
        elif row['SEX_LABEL'] =='Unlassifiable':
            SEX_Total.append(0)
            SEX_Female.append(0)
            SEX_Male.append(0)
            SEX_EMF.append(0)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(1)
        else:
            SEX_Total.append(0)
            SEX_Female.append(0)
            SEX_Male.append(0)
            SEX_EMF.append(0)
            SEX_Classifiable.append(0)
            SEX_Unclassifiable.append(0)

        ##Race grouping of companies 
        Race_list.append(row['RACE_GROUP'])  ##categorical value for NB
        if row['RACE_GROUP_LABEL'] == 'Total': ##changing to binary value for NM
            Race_Total.append(1)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0)
        elif row['RACE_GROUP_LABEL'] == 'White':
            Race_Total.append(0)
            Race_White.append(1)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Black or African American':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(1)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0)
        elif row['RACE_GROUP_LABEL'] == 'American Indian and Alaska Native':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(1)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Asian':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(1)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Native Hawaiian and Other Pacific Islander':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(1)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Minority':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(1)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Equally minority/nonminority':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(1)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Nonminority':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(1)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Classifiable':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(1)
            Race_Unclassifiable.append(0) 
        elif row['RACE_GROUP_LABEL'] == 'Unclassifiable':
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(1) 
        else:
            Race_Total.append(0)
            Race_White.append(0)
            Race_Black.append(0)
            Race_Native.append(0)
            Race_Asian.append(0)
            Race_Islander.append(0)
            Race_Minority.append(0)
            Race_EMN.append(0)
            Race_Nonminority.append(0)
            Race_Classifiable.append(0)
            Race_Unclassifiable.append(0) 
        
        ##Employee Size 
        EmpSiz_list.append(row["EMPSZFI"])
        EmpSiz_Laballist.append(row["EMPSZFI_LABEL"])
        if row["EMPSZFI_LABEL"] == 'Firms with no employees':
            Firms_No.append(1)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 1 to 4 employees':
            Firms_No.append(0)
            Firms_1_4.append(1)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 5 to 9 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(1)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 10 to 19 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(1)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 20 to 49 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(1)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 50 to 99 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(1)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 100 to 249 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(1)
            Firms_250_499.append(0)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 250 to 499 employees':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(1)
            Firms_500.append(0)
        elif row["EMPSZFI_LABEL"] == 'Firms with 500 employees or more':
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(1)
        else:
            Firms_No.append(0)
            Firms_1_4.append(0)
            Firms_5_9.append(0)
            Firms_10_19.append(0)
            Firms_20_49.append(0)
            Firms_50_99.append(0)
            Firms_100_249.append(0)
            Firms_250_499.append(0)
            Firms_500.append(0)
        
        
        
   
        ##OWNER LIST
        QDESC_code = row['QDESC']
        QDESC_num = QDESC_code[1:]
        if len(QDESC_num) == 3:
            QDESC_num1 = QDESC_num[:1]
            Owner_list.append(int(QDESC_num1))
        else:
            Owner_list.append(int(QDESC_num))
        if row['QDESC_LABEL'] =='OWNRNUM':
            QDESC_OWNRNUM.append(1)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(0)
        elif row['QDESC_LABEL'] =='FAMOWN':
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(1)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(0)
        elif row['QDESC_LABEL'] =='SPOUSES':
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(1)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(0)
        elif row['QDESC_LABEL'] =='CUST':
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(1)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(0)
        elif row['QDESC_LABEL'] =='WORKERS':
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(1)
            QDESC_CEASEOPS.append(0)
        elif row['QDESC_LABEL'] =='CEASEOPS':
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(1)
        else:
            QDESC_OWNRNUM.append(0)
            QDESC_FAMOWN.append(0)
            QDESC_SPOUSES.append(0)
            QDESC_CUST.append(0)
            QDESC_WORKERS.append(0)
            QDESC_CEASEOPS.append(0)
        
  
        if row['FIRMPDEMP'] in missing_values:
            Firm_list.append(0)
        else:
            Firm_row = row['FIRMPDEMP']
            Firm_int = int(Firm_row)
            Firm_list.append(Firm_int)
            
        #Employee Size
        if row['EMP'] in missing_values:
            Employee_list.append(0)
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
        elif row['EMP']  == '0 to 19 employees':
            EMP_0_19.append(1)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '20 to 99 employees':
            EMP_0_19.append(0)
            EMP_99.append(1)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)   
            Employee_list.append(0)
        elif row['EMP']  == '100 to 249 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(1)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '250 to 499 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(1)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '500 to 999 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(1)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '1,000 to 2,499 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(1)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '2,500 to 4,999 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(1)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '5,000 to 9,999 employeess':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)                
            EMP_499.append(0)
            EMP_999.apped(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(1)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '10,000 to 24,999 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(1)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '25,000 to 49,999 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(1)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '50,000 to 99,999 employees':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(1)
            EMP_100k.append(0)
            Employee_list.append(0)
        elif row['EMP']  == '100,000 employees or more':
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(1)
            Employee_list.append(0)
        else:
            EMP_0_19.append(0)
            EMP_99.append(0)
            EMP_249.append(0)
            EMP_499.append(0)
            EMP_999.append(0)
            EMP_2499.append(0)
            EMP_4999.append(0)
            EMP_9999.append(0)
            EMP_24k.append(0)
            EMP_50k.append(0)
            EMP_99k.append(0)
            EMP_100k.append(0)
            Employee_list.append(row['EMP'])
            
        if row['PAYANN'] in missing_values:
            Pay_list.append(0)
        else:
            annualpay_int = int(row['PAYANN'])
            Pay_list.append(annualpay_int)
            
        Year_row = row['YEAR']
        Year_int = int(Year_row)
        Year.append(Year_int)
        
        if annualpay_int > 1110000:
            Target.append('2')
        elif  annualpay_int > 1109999 and annualpay_int < 554999 :
            Target.append('1')
        else: 
            Target.append('0')
        


    
KNeighbors_array = np.array((
                            SEX_Total, SEX_Female, SEX_Male, SEX_EMF, SEX_Classifiable,
                            SEX_Unclassifiable,  ##sex binary values 
    
                            #race binary values 
                            Race_Total, Race_White, Race_Black, Race_Native, Race_Asian,
                            Race_Islander, Race_Minority, Race_EMN, Race_Nonminority,
                            Race_Classifiable, Race_Unclassifiable, 
    
                            #Firm size 
                            Firms_No, Firms_1_4, Firms_5_9, Firms_10_19, Firms_20_49,
                            Firms_50_99, Firms_100_249, Firms_250_499, Firms_500,
    
                            #business owners characters
                            QDESC_OWNRNUM, QDESC_FAMOWN, QDESC_SPOUSES, QDESC_CUST,
                            QDESC_WORKERS, QDESC_CEASEOPS,
                          
                            #Employee list
                            EMP_0_19, EMP_99, EMP_249, EMP_499, EMP_999, EMP_2499,
                            EMP_4999, EMP_9999, EMP_24k, EMP_50k, EMP_99k, EMP_100k,
    
                            #Numerical values
                            Firm_list,
                            Pay_list, Year, 
                            Target
)).transpose()
    
    
NB_array = np.array((
                   Sex_list, Race_list, EmpSiz_list, ##categorial values          
                   Owner_list,  Firm_list,  ##numerical values
                  Pay_list, Year, Target)).transpose()
    
print('Dataset Built')





    

female_count = sum(SEX_Female)

male_count = sum(SEX_Male)
    
emf_count = sum(SEX_EMF)
    
white_count = sum(Race_White) 
    
black_ct = sum(Race_Black)
    
native_ct = sum(Race_Native)
    
asian_ct = sum(Race_Asian)
    
islander_ct = sum(Race_Islander)

        
print(female_count, male_count, emf_count, white_count, black_ct, native_ct, asian_ct, islander_ct )       



X = ['Female-owned', 'Male-owned', 'Equally-owned Male/Female', 'White-owned', 'Black-owned', 'Native-owned',
    'Asian-owned', 'Race-owned']
Y = [female_count, male_count, emf_count, white_count, black_ct, native_ct, asian_ct, islander_ct ]

plt.barh(X, Y) 

plt.title("Number of Businesses by Owner Demographics (Figure 1)")
plt.xlabel("Number of Businesses")
plt.ylabel("Owner Demographics")

plt.show()



X = [] 
Y = []

for x in Pay_list:
    X.append(x)
    
    
for y in EmpSiz_Laballist:
    Y.append(y)

fig, ax = plt.subplots()

plt.title("Business Size and Annual Pay (Figure 2)")
plt.xlabel("Annual Pay in $ (m for millions, b for billions)")
plt.ylabel("Business Size")


ax.set_xticks([10000000, 500000000, 1000000000, 2000000000, 3000000000])

ax.set_xticklabels(( '10m', '500m', '1b', '2b', '3b'))

plt.scatter(X,Y) 

plt.show()



##KNeighbors
#Spilt into test/train datasets 

KN_X = KNeighbors_array[:, :-1] # all but last column for X 
KN_X = KN_X.astype(np.float64)


KN_Y = KNeighbors_array[:, -1] # last column for classification target, Y 

KNX_train, KNX_test, KNY_train, KNY_test = train_test_split(KN_X, KN_Y, test_size=0.25, random_state=42)





##prevent future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##Machine learning algorithms: KNN



neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(KNX_train, KNY_train)
KNY_pred = neigh.predict(KNX_test)




print('Error Rate of KNN:', np.mean(KNY_pred != KNY_test))



##prevent future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##Machine learning algorithms: KNN

K_list = []
K_error = []

for k in [1,2,3, 4, 5]:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(KNX_train, KNY_train)
    KNY_pred = neigh.predict(KNX_test)
    K_list.append(k)
    K_error.append(np.mean(KNY_pred != KNY_test))
    print('Error Rate of KNN:', k,  np.mean(KNY_pred != KNY_test))





fig, ax = plt.subplots()


ax.set_xticks([1, 2, 3, 4, 5])


plt.title("Error Rates for K-Values (Figure 3)")
plt.xlabel("K-Value")
plt.ylabel("Error Rate")
plt.plot(K_list,K_error) 
plt.show()




#Naive Bayes 
#Spilt into test/train datasets 
##will have to be aware which columns are where 

NB_X = NB_array[:, :-1] # all but last column for X 



NB_Y = NB_array[:, -1] # last column for classification target, Y 

NBX_train, NBX_test, NBY_train, NBY_test = train_test_split(NB_X, NB_Y, test_size=0.25, random_state=42)






CatColsTest = NBX_test[:, :-6]      
CatColsTrain = NBX_train[:, :-6]

Cclf = CategoricalNB()
Cclf.fit(CatColsTrain, NBY_train) 

CNBY_pred = Cclf.predict(CatColsTest)
NBY_pred_prob = Cclf.predict_proba(CatColsTest)

CategoricalNB_error = np.mean(CNBY_pred != NBY_test)
##mutiply means of probabilibilies 

        
NBY_pred_prob_mean = np.mean(NBY_pred_prob[0] * NBY_pred_prob[1])

print('Error Rate of CategoricalNB:', CategoricalNB_error)
print('Probabilty' ,NBY_pred_prob_mean )



##Machine learning algorithms: Naive Bayes
##sklearn.naive_bayes.CategoricalNB for categorical values 
##clf = GaussianNB() for numerical values 

        
NB_X = NB_array[:, :-1] # all but last column for X 
NB_X = NB_X.astype(np.int)


NB_Y = NB_array[:, -1] # last column for classification target, Y 

NBX_train, NBX_test, NBY_train, NBY_test = train_test_split(NB_X, NB_Y, test_size=0.25, random_state=42)
      

NumColsTest = NBX_test[:,-4:] 
NumColsTest.astype(np.int)
NumColsTrain = NBX_train[:,-4:] 
NumColsTrain.astype(np.int)

        
Gclf = GaussianNB()
Gclf.fit(NumColsTrain, NBY_train)

GBY_pred = Gclf.predict(NumColsTest)


GaussianNB_error = np.mean(GBY_pred  != NBY_test)


GBY_pred_prob = Gclf.predict_proba(NumColsTest)

GBY_pred_prob_mean = np.mean(GBY_pred_prob[0] * GBY_pred_prob[1])


print('Error Rate of GaussianNB:', GaussianNB_error)
print('Probabilty:' ,GBY_pred_prob_mean )



NB_mean_error = (CategoricalNB_error*GaussianNB_error)/2
print('Error Rate of NB', NB_mean_error)


NB_prob = (NBY_pred_prob_mean * GBY_pred_prob_mean)
print('Probabilty of NB:',NB_prob)








