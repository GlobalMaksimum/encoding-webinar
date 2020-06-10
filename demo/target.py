import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.title("Target (Mean) Encoding")

    st.markdown(" * In Target Encoding for each category in the feature label is decided with the mean value of the target variable on a training data.")

    image = Image.open('images/target.png')
    st.image(image, use_column_width=True)

    st.info(":pushpin:  This encoding method brings out the relation between similar categories.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])
    
    st.warning(":exclamation: First we need to convert target categories to numeric values.")
    
    st.subheader("Simply replace")
    with st.echo():
        df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
    
    targetSum = df.groupby(feat)['target'].agg('sum')
    targetCount = df.groupby(feat)['target'].agg('count')
    targetEnc2 = targetSum/targetCount

    steps = pd.DataFrame(targetSum)
    steps = pd.concat([steps,targetCount], axis=1)
    steps = pd.concat([steps,targetEnc2], axis=1)
    steps.columns = ['Sum of Targets','Count of Targets','Mean of Targets']
    
    targetEnc2 = dict(targetEnc2)
    X_target = pd.DataFrame(df[feat])
    X_target[f'tranformed_{feat}'] = df[feat].replace(targetEnc2).values
    
    button = st.button('Apply Target Encoding')
    if button:
        st.subheader("Steps for calculating Mean of Targets")
        st.dataframe(steps)
        st.dataframe(X_target)

    st.warning(":exclamation: Because we don't know targets of the test data this can lead the overfitting.")
    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        st.subheader("`category_encoders` has a module for Target Encoding")
        with st.echo():
            import category_encoders as ce
            targetEnc = ce.TargetEncoder()
            X_target[f'tranformed_{feat}'] = targetEnc.fit_transform(df[feat],df['target'])
    
    st.subheader("Label Encoding vs Target Encoding")
    
    st.markdown("* Target encoding can be viewed as a variation of label encoding, that is used to make labels correlate with the target.")
        
    df_all = pd.DataFrame(df[feat])
    df_all.columns = ['feature']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_all[f'feature_label'] = le.fit_transform(df_all['feature'])
    df_all[f'feature_mean'] = df_all['feature'].replace(targetEnc2).values
    df_all['target'] = df['target']
    st.dataframe(df_all)
    
    #df_all.groupby("target").feature_label.hist(alpha=0.4)
    #st.pyplot()

    #df_all.groupby("target").feature_mean.hist(alpha=0.4)
    #st.pyplot()
    
    st.info(":pushpin:  Label encoding gives random order. No correlation with target.")
    st.info(":pushpin:  Mean encoding helps to separate zeros from ones.")

    def sephist(col):
        target_1 = df_all[df_all['target'] == 1][col]
        target_0 = df_all[df_all['target'] == 0][col]
        return target_1, target_0

    for num, alpha in enumerate(['feature_label','feature_mean']):
        plt.subplot(1, 2, num+1)
        plt.hist((sephist(alpha)[0], sephist(alpha)[1]), alpha=0.5, label=['target_1', 'target_0'], color=['r', 'b'])
        #plt.hist(sephist(alpha)[0], alpha=0.5, label='target_1', color='b')
        #plt.hist(sephist(alpha)[1], alpha=0.5, label='target_0', color='r')
        plt.legend(loc='upper right')
        plt.title(alpha)
    plt.tight_layout(pad=1)
    st.pyplot()

    st.markdown(" * The plot looks kind of sorted, this sorting quality of mean encoding is quite helpful.")
    
    '''
    image2 = Image.open('images/target1.png')
    st.image(image2, use_column_width=True)
    
    st.markdown(" * Label encoding will be hard to process for a model, because of how random it will look and how many splits will be needed.")
    st.markdown(" * Trees have limited depth, with mean encoding we can compansate it.")
    
    image3 = Image.open('images/target2.png')
    st.image(image3, use_column_width=True)
    
    st.markdown(" * As seen above, with increasing the depths of trees training score and also validation score becomes better.")
    st.markdown(" * So trees need a huge number of splits to extract information from some variables.")
    #st.markdown(" * Some features havea tremendous amount of split ponts.")
    st.markdown(" * Mean encoding helps model to treat all these categories differently.")
    
    image4 = Image.open('images/target3.png')
    st.image(image4, use_column_width=True)
    
    st.markdown(" * As seen above, while training 'roc_auc_score' is nearly 1, validation 'roc_auc_score' around 0.55.")
    st.markdown(" * It is a clear sign of overfitting.")
    '''
    
    st.info(":pushpin:  A theoretical basis for this approach was given in the classic paper [1].")
    
    showImplementation2 = st.checkbox('Details', key='key2') 
    
    if showImplementation2:
        st.info(":pushpin:  The key transformation used in the proposed scheme is one that maps each instance of a high-cardinality categorical to the probability estimate of the target attribute.")
        st.subheader("Binary Target")
        st.markdown('''In a classification scenario, the numerical representation corresponds to the posterior probability of the target, conditioned by the value of the categorical attribute.''')

        st.latex("X_i \longrightarrow S_i \cong P(Y|X=X_i)")

        st.markdown('''In reality this means that we compute mean for the target variable for each category and encode that category with the target mean.''')

        st.latex(r"S_i = \frac{n_{iY}}{n_i}")

        st.warning(":exclamation: One problem of target encoding is that some of the categories have few training examples, and the mean target value for these categories may assume extreme values, so encoding these values with mean may reduce the model performance.")
        
        st.markdown("To deal with this issue the target mean for the category is often mixed with the marginal mean of the target variable [1]:")

        st.latex(r"S_i = \lambda(n_i)\frac{n_{iY}}{n_i}+(1-\lambda(n_i))\frac{n_Y}{n_{TR}}")

        st.markdown('''The weights λ are close to one for the categories with many training examples and close to zero for rare categories. For example, it can be parametrized as:''')

        st.latex(r"\lambda(n)=\frac{1}{1+e^{-\frac{(n-k)}{f}}}")
        
        st.subheader("Continuous Target")
       
        st.markdown("The preprocessing scheme proposed for binary targets can also be applied to the case of a continuous target attribute $$Y$$.")
        st.markdown("In a prediction scenario, the numerical representation corresponds to the expected value of the target given the value of the categorical attribute.")
        st.markdown("Although there is a significant difference between estimating expected values and estimating probabilities, the formula in third equation remains basically unchanged. However, now we consider the average of $$Y$$ across the training data, $$E[Y]$$ as the 'null hypothesis', and the average of $$Y$$ when $$X=X_i$$ as the raw estimate for the cell [1]:")
        
        st.latex(r"S_i = \lambda(n_i)\frac{\sum_{k \epsilon L_i}Y_k}{n_i}+(1-\lambda(n_i))\frac{\sum_{k=1}^{N_{TR}} Y_k}{n_{TR}}")
        
        st.markdown("The principle is to weight the estimate of the target toward $$E[Y|X=X_i]$$ when the sample size is large, and toward the (training) population average when the sample size is small.")

        #st.markdown(" ### Bayesian Target Encoding")

        #st.latex(r"P(\Theta|y)=\frac{P(y|\Theta)P(\Theta)}{P(y)}")

        #st.markdown('''In a prediction scenario, the numerical representation corresponds to the expected value of the target given the value of the categorical attribute.''')
    
    st.warning(":exclamation: Because we don't know targets of the test data this can lead the overfitting.")
    st.warning(":exclamation: Target encoding also introduces noise (comes from the noise in the target variable) into the encoding of the categorical variables.")
    
    st.info(" [1] Micci-Barreca, Daniele. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.. SIGKDD Explorations. 3. 27-32. 10.1145/507533.507538.")
    
''' import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    
    X_train, X_test, y_train, y_test = train_test_split(df_all[['feature_mean']], df_all[['target']], test_size=65)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=48)
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val)
    
    loss_mean = []
    
    for i in range(1,6):
        params = {
            'objective' :'binary',
            'learning_rate' : 0.02,
            'num_leaves' : 76,
            'feature_fraction' : 0.64, 
            'bagging_fraction' : 0.8, 
            'bagging_freq' : 1,
            'boosting_type' : 'gbdt',
            'metric' : 'binary_logloss',
            'max_dept' : i
        }
        model = lgb.train(params, train_set, valid_sets = [train_set, val_set])
        y_pred = model.predict(X_test)
        #model = lgb.LGBMClassifier(objective = 'binary')
        #model.fit(X_train, y_train)
        #st.dataframe(pd.DataFrame(y_test))
        #st.dataframe(pd.DataFrame(y_pred))
        #score = metrics.log_loss(y_pred, y_test)
        #loss_mean.append(score)
    
    X_train, X_test, y_train, y_test = train_test_split(df_all[['feature_label']], df_all[['target']], test_size=65)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=48)
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val)
    
    loss_label = []
    
    for i in range(1,6):
        params = {
            'objective' :'binary',
            'learning_rate' : 0.02,
            'num_leaves' : 76,
            'feature_fraction' : 0.64, 
            'bagging_fraction' : 0.8, 
            'bagging_freq' : 1,
            'boosting_type' : 'gbdt',
            'metric' : 'binary_logloss',
            'max_dept' : i
        }
        model = lgb.train(params, train_set, valid_sets = [train_set, val_set])
        y_pred = model.predict(X_test)
        #score = metrics.roc_auc_score(y_pred, y_test)
        #loss_label.append(score)'''
