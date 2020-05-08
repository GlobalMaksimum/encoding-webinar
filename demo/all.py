import streamlit as st
import pandas as pd
import numpy as np 
import altair as alt
from PIL import Image
import pydoc

st.subheader('Data Import')

st.markdown('* Dataset download source from [OpenML Breast cancer](https://www.openml.org/d/13)')
st.markdown('* This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia.')
st.markdown('* ***Number of Instances:*** 286')
st.markdown('* ***Number of Features:*** 9')
st.markdown('* ***Target Class:*** Whether the tumor will recur or not.')
st.markdown("""
                * Data fields (Features)  
                    * **age:** Age (in years at last birthday) of the patient at the time of diagnosis 
                    * **menopause:** Whether the patient is pre- or postmenopausal at time of diagnosis       
                    * **tumor-size:** The greatest diameter (in mm) of the excised tumor       
                    * **inv-nodes:** The number (range 0 - 39) of axillary lymph nodes that contain metastatic breast cancer visible on histological examination         
                    * **node-caps:** If the cancer does metastasise to a lymph node, although outside the original site of the tumor it may remain “contained” by the capsule of the lymph node. However, over time, and with more aggressive disease, the tumor may replace the lymph node and then penetrate the capsule, allowing it to invade the surrounding tissues   
                    * **deg-malig:**  Degree of malignancy
                    * **breast:** Breast cancer may obviously occur in either breast
                    * **breast-quad:** The breast may be divided into four quadrants, using the nipple as a central point
                    * **irradiat:**  Radiation therapy is a treatment that uses high-energy x-rays to destroy cancer cells.    
                """)  

with st.echo():
    df = pd.read_csv('../data/breast-cancer.csv')

st.dataframe(df)

st.subheader('Simple Exploration')

feat = st.selectbox('Select Feature',('age','menopause','tumor-size', 
                                      'inv-nodes','node-caps','deg-malig',
                                      'breast','breast-quad','irradiat'))

bar1 = alt.Chart(df).mark_bar().encode(
    x = alt.X(feat),
    y = 'count()',
    color = 'Class')

st.altair_chart(bar1,use_container_width=True)

st.markdown("Define target for our classification problem.")

with st.echo():
    df.rename(columns={'Class': 'target'}, inplace=True)

df.fillna('unknown', inplace = True)

st.title("A Quick Look at Encoding")

st.markdown("""
         Encoding is the process of converting the data or a given sequence of characters, symbols, alphabets etc., into a specified format, for the secured transmission of data.
        """)

st.header('We Usually Have Categorical Variables in Our Dataset')

st.markdown('A categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values.')

st.header("Categorical Variables can be divided into 2 categories:")

image1 = Image.open('../images/cat.png')

st.image(image1, use_column_width=True)

st.markdown('Nominal variable that has no numerical importance, besides ordinal variable has some order.')

st.header('Why do we need encoding for Categorical Variables')
 
st.markdown("""
            Because majority of algorithms are implemented using Linear Algebra primitives. Such as
            * Logistic Regression
            * SVM
            * LightGBM
    """)

st.subheader('Example scikit-learn API')

with st.echo():
    from sklearn.linear_model import LogisticRegression

with st.echo():
    strhelp = pydoc.render_doc(LogisticRegression.fit)


strhelp = pydoc.render_doc(LogisticRegression.fit)
st.markdown(strhelp)


st.subheader('A non-rigirous: How to convert numeric data into non numeric strings')

feats = st.multiselect('Select a couple features',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

if feats:
    st.dataframe(df[feats])

st.subheader('Fail to Build a LogisticRegression Model using String Values')

dt = st.selectbox('Select Data Type:', ('string','somethig else'))

if dt=='string':
    st.markdown("""
                    ```python
                    lr = LogisticRegression()
                    lr.fit(df[feats],df.Class)
                    ```
        """)
    with st.echo():
        lr = LogisticRegression()
        lr.fit(df[feats],df.Class)

else:
    st.subheader('Simplest Idea')
    st.markdown(""" 
    What if I simply replace string value with arbitrarily picked unique values (aka dictionary mapping)
        * Unique value per column/feature/attribute
        """)

    with st.echo():
        X_enc = df[['breast', 'irradiat']].replace({'right':1,'left':0, 'no':0,'yes':1})
        y_enc = df.target.replace( {'recurrence-events':1,'no-recurrence-events':0 } )

        lr = LogisticRegression().fit(X_enc, y_enc)

st.success('IT WORKED!!')
st.markdown("""
    * Can we have better encoding techniques ?
    * That's why we are here
    """)
    
df['breast'].replace({'right': 1, 'left': 0}, inplace=True)
df['irradiat'].replace({'yes': 1, 'no': 0}, inplace=True)
df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)

st.header('How to Generalize Dictionary Encoding into N distrinct values ?')

feat = st.selectbox('Select Feature',(
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

st.dataframe(df[f'{feat}'])

with st.echo():
    df[feat] =  df[feat].fillna('unknown')
    df[feat].unique()

st.markdown('Unique values are')
st.write(df[feat].unique())



st.subheader('`sklearn` has a great module for this')

with st.echo():
    from sklearn.preprocessing import LabelEncoder  
        
with st.echo():
    le = LabelEncoder()

button = st.button('fit_transform')

if button:
    with st.echo():
        le.fit_transform(df[feat])
    st.write(le.fit_transform(df[feat]))

st.markdown("""
    What are the potential issues with LabelEncoder approach ?
    """)


st.markdown("""
    ## One Hot Encoding
    1. Generate columns/attributes/features as many as the number of distrinct values in encoded column/attribute/feature
    2. Set only 1 relevant column/attribute/feature value to 1 and 0 others in the encoded domain
    """)

image1 = Image.open('../images/ohe2.png')
st.image(image1, use_column_width=True)

feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast-quad','irradiat'))

st.dataframe(df[f'{feat}'])

with st.echo():
    pd.get_dummies(df[feat])

st.write(pd.get_dummies(df[feat]))

st.markdown(""" 
    ### Why?
    - How similar/dissimilary each value in breast-quad with respect to each other ?
    - Note that the answer of question is mainly related with the encoding you use.
    """)

st.markdown('### 1. Similarity/Dissimilarity each value in **breast-quad** wrt each other by LabelEncoder')
with st.echo():
    df['breast-quad'] =  df['breast-quad'].fillna('unknown')
    unq_values = df['breast-quad'].unique()
    unq_values

with st.echo():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    enc_le = le.fit(df['breast-quad']).transform(unq_values).reshape(-1,1)
st.write(enc_le)

st.markdown('#### Calculate Pairwise Distance Between Encoding of Each Unique Value')
with st.echo():
    from  sklearn.metrics import pairwise_distances
    pairwise_distances(enc_le.reshape(-1,1))
st.write(pairwise_distances(enc_le.reshape(-1,1)))



st.markdown('### 2. Similarity/Dissimilarity each value in breast-quad wrt each other by our new encoding scheme')
with st.echo():
    from sklearn.preprocessing import OneHotEncoder
    ohe= OneHotEncoder()
    enc_ohe = ohe.fit(df[['breast-quad']]).transform(unq_values.reshape(-1,1)).toarray()
    enc_ohe

st.subheader('Calculate Pairwise Distance Between Each Unique Value')
with st.echo():
    pairwise_distances(enc_ohe)

st.write(pairwise_distances(enc_ohe))

st.error('3D VIS WILL COME HERE')

st.markdown('Hence our new encoding preserves relative similarity/dissimilarity of each unique value.')

st.header('What if our string feature has an Alphanumeric Order?')

with st.echo():
    'A' > 'B'
    'A' > 'a'
    '42' > '13'

st.markdown("""
        There is an order between the string values that might need preserving.
         * Let's check our data for a such case.
        
        """)

feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps',
                                    'breast','breast-quad','irradiat'))

st.dataframe(df[f'{feat}'])

if feat=='age':
    st.markdown("""
    `Age` seems like to display such order.
    """ )
elif feat=='inv-nodes':
    st.markdown("""
        So does `inv-nodes`. 
        """ )
elif feat=='tumor-size':
    st.markdown("""
        So does `tumor-size`. 
        """ )

st.header('Does Label Encoder Infer and Preserve the Relationship?')

with st.echo():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    unq_values = df[feat].unique()
    unq_values


button = st.button('fit_transform label enc')

if button:
    enc_le = le.fit(df[feat]).transform(unq_values).reshape(-1,1)
    st.subheader('Mapped Values')
    st.write(np.hstack([unq_values.reshape(-1,1),enc_le]))
    st.subheader('Pairwise Distances')
    with st.echo():
        from  sklearn.metrics import pairwise_distances
        pairwise_distances(enc_le.reshape(-1,1))
    st.write(pairwise_distances(enc_le.reshape(-1,1)))

    st.markdown('Encoded order is arbitrary. But we know our specific order...')


st.header('Can we inject the a specific order information to our encoder.')

with st.echo():
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder(categories=[['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44', '45-49','50-54']])


button2 = st.button('fit_transform ordinal')

if button2:
    enc_oe = oe.fit(df[[feat]]).transform(df[[feat]].drop_duplicates()).reshape(-1,1)
    st.subheader('Mapped Values')
    mapped = np.hstack([unq_values.reshape(-1,1),enc_oe])
    mapped_df = pd.DataFrame({'0':mapped[:,0],'1':mapped[:,1]})
    sorted_df = mapped_df.sort_values('1')
    sorted_df['1'] = sorted_df['1'].astype(np.int8)
    st.write(sorted_df)
    st.subheader('Pairwise Distances')
    with st.echo():
        from sklearn.metrics import pairwise_distances

    st.write(pairwise_distances(sorted_df['1'].values.reshape(-1,1)))

st.header("Thermometer (Unary) Encoding")

image2 = Image.open('../images/thermo.png')
st.image(image2, use_column_width=True)

st.markdown("For ordinal features by applying thermometer encoding we can preserve the order.")

feat = st.selectbox('Select Feature',('age','tumor-size',
                                    'inv-nodes','deg-malig'))

X_thermo = pd.DataFrame(df[feat])
    
st.dataframe(X_thermo.head(10))

from sklearn.base import TransformerMixin
from itertools import repeat
import scipy

class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result

st.markdown("As in ordinal encoding we need to give order between categories.")
    
with st.echo():
    thermos = []
        
    if feat == 'age':
        sort_key = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'].index
        
    elif feat == 'tumor-size':
        sort_key = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'].index
        
    elif feat == 'inv-nodes':
        sort_key = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26'].index
        
    elif feat == 'deg-malig':
        sort_key = int
        
    else:
        raise ValueError(feat)
    
    thermoEnc = ThermometerEncoder(sort_key = sort_key)
    thermos.append(thermoEnc.fit_transform(X_thermo[feat]))

thermo_ohc = scipy.sparse.hstack(thermos).tocsr()
thermo_ohc = scipy.sparse.csr_matrix(thermo_ohc).todense()
X_thermo = pd.DataFrame(thermo_ohc)
    
st.dataframe(df[feat])

button5 = st.button('Apply Thermometer Encoding')
if button5:
    st.dataframe(X_thermo)

#st.header("We Can Reduce Dimensionality Using Binary Encoding")

st.header("Binary Encoding")

st.markdown("Binary encoding converts a category into binary digits. Each binary digit creates one feature column.")

image3 = Image.open('../images/binary.png')
st.image(image3, use_column_width=True)

st.markdown("Compared to one hot encoding, this will require fewer feature columns.")

import category_encoders as ce
    
#st.markdown("Here we apply binary encoding for ordinal columns.")
    
feat2 = st.selectbox('Select Feature',('tumor-size',
                                    'inv-nodes','deg-malig'))

X_binary = pd.DataFrame(df[feat2])
    
st.dataframe(X_binary)

#X_binary[['deg-malig']] = X_binary[['deg-malig']].astype('object')

binaryEnc = ce.BinaryEncoder()
X_binary = binaryEnc.fit_transform(X_binary)

button3 = st.button('Apply Binary Encoding')
if button3:
    st.dataframe(X_binary)

st.header("BaseN Encoding")

#st.header("We Can Reduce Dimensionality Even More by Applying BaseN Encoding")

st.markdown("BaseN encoding converts a category into base N digits.")

st.markdown("By reducing dimensionality, we also lose some information.")

st.markdown("Here we apply base 3 encoding for ordinal columns.")

feat3 = st.selectbox('Select Feature',('tumor-size',
                                    'inv-nodes',))

X_baseN = pd.DataFrame(df[feat3])
    
st.dataframe(X_baseN)

#X_baseN[['deg-malig']] = X_baseN[['deg-malig']].astype('object')


baseNEnc = ce.BaseNEncoder(base = 3)
X_baseN = baseNEnc.fit_transform(X_baseN)

button4 = st.button('Apply BaseN Encoding')
if button4:
    st.dataframe(X_baseN)

st.header("Frequency Encoding")

#st.markdown("It is a way to utilize the frequency of the categories as labels.")

image1 = Image.open('../images/freq.jpeg')
st.image(image1, use_column_width=True)

st.markdown("In the cases where the frequency is related somewhat with the target variable, it helps the model to understand and assign the weight in direct and inverse proportion, depending on the nature of the data.")

feat2 = st.selectbox('Select Feature',('menopause','tumor-size', 
                                      'inv-nodes','node-caps','deg-malig',
                                      'breast','breast-quad','irradiat'))
    
X_freq = pd.DataFrame(df[feat2])
    
st.dataframe(X_freq)

freqEnc = (X_freq.groupby(feat2).size()) / len(X_freq)
freqEnc2 = pd.DataFrame(freqEnc)
freqEnc2.columns = ['Frequency']

button1 = st.button('Frequencies of Categories')
if button1:
    st.dataframe(pd.DataFrame(freqEnc2))
    
X_freq[feat2] = X_freq[feat2].apply(lambda x : freqEnc[x])
    
button2 = st.button('Apply Frequency Encoding')
if button2:
    st.dataframe(X_freq)

st.header("Target (Mean) Encoding")

st.markdown("In Target Encoding for each category in the feature label is decided with the mean value of the target variable on a training data.")

image2 = Image.open('../images/target.png')
st.image(image2, use_column_width=True)

st.markdown("This encoding method brings out the relation between similar categories.")

feat3 = st.selectbox('Select Feature',('tumor-size', 
                                      'inv-nodes','node-caps','deg-malig',
                                      'breast','breast-quad','irradiat'))

st.dataframe(df[feat3])

X_target_nom = pd.DataFrame(df[[feat3,'target']])

targetSum = X_target_nom.groupby(feat3)['target'].agg('sum')
targetCount = X_target_nom.groupby(feat3)['target'].agg('count')
targetEnc = targetSum/targetCount

steps = pd.DataFrame(targetSum)
steps = pd.concat([steps,targetCount], axis=1)
steps = pd.concat([steps,targetEnc], axis=1)
steps.columns = ['Sum_of_Targets','Count_of_Targets','Mean_of_Targets']

button3 = st.button('Steps for Target Encoding')
if button3:
    st.dataframe(steps)

targetEnc = dict(targetEnc)
X_target_nom[feat3] = X_target_nom[feat3].replace(targetEnc).values

X_target_nom.drop(['target'], axis=1, inplace=True)

button4 = st.button('Apply Target Encoding')
if button4:
    st.dataframe(X_target_nom)

st.markdown("Because we don't know targets of the test data this can lead the overfitting.")

st.header("K-Fold Target Encoding")

st.markdown("By dividing the data into folds we can reduce the overfitting.")

image3 = Image.open('../images/kfold.png')
st.image(image3, use_column_width=True)

from sklearn.model_selection import KFold

feat4 = st.selectbox('Select Feature',( 'inv-nodes','node-caps','deg-malig',
                                      'breast','irradiat'))

st.dataframe(df[feat4])

X_kfold_nom = pd.DataFrame(df[[feat4,'target']])

kf = KFold(n_splits = 5, shuffle = False)
fold_df = pd.DataFrame()

for train_ind,val_ind in kf.split(X_kfold_nom):
    if(X_kfold_nom[feat4].dtype == 'object') and 'tenc' not in feat4:
        replaced = X_kfold_nom.iloc[train_ind][[feat4,'target']].groupby(feat4)['target'].mean()
        fold_df = pd.concat([fold_df, pd.DataFrame(replaced)], axis = 1)
        replaced = dict(replaced)
        X_kfold_nom.loc[val_ind,f'tenc_{feat4}'] = X_kfold_nom.iloc[val_ind][feat4].replace(replaced).values

fold_df.columns = ['fold_1','fold_2','fold_3','fold_4','fold_5']

button5 = st.button('Mean of Targets in K-Fold')
if button5:
    st.dataframe(fold_df)

X_kfold_nom.drop([feat4,'target'], axis=1, inplace=True)

button6 = st.button('Apply K-Fold Target Encoding')
if button6:
    st.dataframe(X_kfold_nom)

st.header("Weight of Evidence Encoding")

st.markdown("Weight of Evidence (WoE) is a measure of the “strength” of a grouping technique to separate good and bad.")

image4 = Image.open('../images/woe3.jpg')
st.image(image4, use_column_width=True)

st.markdown("Weight of evidence (WOE) is a measure of how much the evidence supports or undermines a hypothesis.")

image5 = Image.open('../images/woe.jpg')
st.image(image5, use_column_width=True)

st.markdown("This method was developed primarily to build a predictive model to evaluate the risk of loan default in the credit and financial industry.")

st.markdown("WoE is well suited for Logistic Regression because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)).")

from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder

feat5 = st.selectbox('Select Feature',('node-caps','deg-malig',
                                      'breast','breast-quad','irradiat'))

st.dataframe(df[feat5])

X_woe_nom = pd.DataFrame(df[feat5])

weightEnc = WeightOfEvidenceEncoder()
X_woe_nom = weightEnc.fit_transform(X_woe_nom,df['target'])

button7 = st.button('Apply Weight of Evidence Encoding')
if button7:
    st.dataframe(X_woe_nom)

#X_woe_ord[['deg-malig']] = X_woe_ord[['deg-malig']].astype('object')

df = pd.read_csv('../data/train.csv')

X_cyclic = df[['day','month']].copy()

st.title("Cyclic Encoding")

st.markdown("""
        The transformation of cyclic features is important because when cyclic features are untransformed then there's no way for the model to understand that the smallest value in the cycle is actually next to the largest value.
        """)

st.header('What is Cyclic Encoding?')

st.markdown("""
         The main idea behind cyclic encoding is to enable cyclic data to be represented on a circle.
        """)

image = Image.open('../images/cyclic2.png')

st.image(image)

st.markdown("""
         Examples are 'day' and 'month' features.
        """)

st.dataframe(X_cyclic)

st.header("Let's Map These Features onto The Unit Circle As in Below.")

with st.echo():
    X_cyclic['day_sin'] = np.sin(2 * np.pi * X_cyclic['day']/7)
    X_cyclic['day_cos'] = np.cos(2 * np.pi * X_cyclic['day']/7)
    X_cyclic['month_sin'] = np.sin(2 * np.pi * X_cyclic['month']/12)
    X_cyclic['month_cos'] = np.cos(2 * np.pi * X_cyclic['month']/12)

X_cyclic.drop(['day','month'], axis=1, inplace=True)

button = st.button('Apply Cyclic Encoding')
if button:
    st.dataframe(X_cyclic)

X_label = pd.read_csv('../data/X_label.csv')
X_ord = pd.read_csv('../data/X_ord.csv')
X_ohe_nom = pd.read_csv('../data/X_ohe_nom.csv')
X_ohe_ord = pd.read_csv('../data/X_ohe_ord.csv')
X_binary = pd.read_csv('../data/X_binary.csv')
X_base3 = pd.read_csv('../data/X_base3.csv')
X_thermo = pd.read_csv('../data/X_thermo.csv')

#st.header('Now We Can Construct Logistic Regression Model for Classification')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

def logistic(X,y):
    model = LogisticRegression(C = 0.12345678987654321, solver = "lbfgs", max_iter = 5000, tol = 1e-2, n_jobs = 48)
    model.fit(X, y)
    score = cross_validate(model, X, y, cv=3, scoring="roc_auc")["test_score"].mean()
    print('AUC Score: ',f"{score:.6f}")
    