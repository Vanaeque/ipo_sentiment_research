import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd

def smart_log(X):
    return np.sign(X)*np.log(np.abs(X) + 1)

def smart_sqrt(X):
    return np.sign(X)*np.sqrt(np.abs(X))


def null_drop(data, cutoff = 0.5):
    
    n = data.shape[0]
    cols = list(data.columns)
    
    for col in data.columns:
        n_nulls = data[col].isnull().sum()
        if n_nulls/n > cutoff:
            cols.remove(col)
            
    return cols
            
def nonsense_drop(data, cutoff = 0.5):
    
    n = data.shape[0]
    cols = list(data.columns)
    
    for col in data.columns:
        
        n_top = data[col].value_counts().sort_values(ascending=False).iloc[0]
        if n_top/n > cutoff:
            cols.remove(col)
            
    return cols

def corr_drop(X, y, cutoff = 0.5):
    
    fl = True
    cols = list(X.columns)
    
    while fl:
        fl = False
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col1=cols[i]
                col2=cols[j]
                r2 = abs(spearmanr(X[col1], X[col2])[0])
                if r2 > cutoff:
                    fl = True
                    if spearmanr(X[cols[i]], y)[0] > spearmanr(X[cols[j]], y)[0]:
                        cols.remove(cols[i])
                    else:
                        cols.remove(cols[j])
                if fl:
                    break
            if fl:
                break
                
    return cols


def linearization(X, y):

    funcs = [np.square, np.sqrt, np.log, smart_log, smart_sqrt]
    
    cols = X.columns
    res_funcs = []
    
    for col in cols:
        r2 = spearmanr(X[col], y)
        best_func = None
        for func in funcs:
            try:
                X_ = func(X[col])
                r2_ = spearmanr(X_, y)
                if r2_ > r2:
                    r2 = r2_
                    best_func = func
                    
            except Exception as e: 
                print(e)
        
        res_funcs.append(func)
                 
    return res_funcs


def woe_line(X, y, n_buckets = 20, feature_name = 'feature', target_name = 'target', clip = True):
    
    """ Строит график зависимости WoE
    x - параметр, от которого стоит искать зависимость
    y - метки класса (0 / 1)
    n_buckets - количество бинов для вещественного признака
    feature_name, target_name - подписи к графику
    """
    
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import roc_auc_score
    
    def make_bucket(feature, n=100):
        '''функция, бьющая на бакеты(по умолчанию 100 точек)'''
        return df.assign(bucket = np.ceil(feature.rank(pct=True) * n))
    
    def WoE (df_1, feature_name = 'feature', target_name = 'target'):

        a_1 = df_1[df_1 == 1].count()
        b_1 = df_1[df_1 == 0].count()

        a = df[df[target_name] == 1].count()[feature_name]
        b = df[df[target_name] == 0].count()[feature_name]

        c = np.log(a_1/b_1) - np.log(a/b)

        return c

    def p_diff(df_1, feature_name = 'feature', target_name = 'target'):

        a_1 = df_1[df_1 == 1].count()
        b_1 = df_1[df_1 == 0].count()

        a = df[df[target_name] == 1].count()[feature_name]
        b = df[df[target_name] == 0].count()[feature_name]

        c = a_1/a - b_1/b

        return c
     
        
    df = pd.DataFrame({feature_name : X, target_name : y})
    df = df[df[feature_name].notna()]
    
    df_buckets = make_bucket(df[feature_name], n_buckets)
    
    df_info = df_buckets.groupby('bucket').agg(
            feature_value = pd.NamedAgg(column = feature_name, aggfunc='mean'),
            target_value =  pd.NamedAgg(column = target_name, aggfunc = lambda a : WoE(a, feature_name)),
            p_diff = pd.NamedAgg(column = target_name, aggfunc = lambda a : p_diff(a, feature_name)))
    
    fig = go.Figure(data=go.Scatter(
            name = 'Initial data',
            y=df_info['target_value'],
            x=df_info['feature_value'],
            line = (dict(
                color = 'indianred',
                dash = 'dot')),
            error_y=dict(
                type='constant',
                value = 0.05,
                symmetric=True,
                color = 'indianred')
            ))
    
    
    model = LinearRegression()

    X = np.array(df_info['feature_value']).reshape(-1, 1)
    y = np.array(df_info['target_value'])
    model.fit(X = X, y = y)
    x = np.linspace(np.min(df_info['feature_value']), np.max(df_info['feature_value']), 100)
    r2 = np.round(r2_score(df_info['target_value'], model.predict(np.array(df_info['feature_value']).reshape(-1,1))), 3)
    

    x_0 = np.round(- model.intercept_/model.coef_[0], 3)
    
    auc = np.round(roc_auc_score(df[target_name], df[feature_name]), 3)
    
    iv = np.round((df_info['p_diff']*df_info['target_value']).sum(), 3)
    
    
    fig.add_trace(go.Scatter(
                name = 'Regression',
                x = x,
                y = model.predict(np.array(x).reshape(-1,1)),
                line = dict(
                color = 'blue'
                )))
    
    fig.update_layout(
        title_text = f' {feature_name} | AUC = {auc} | IV = {iv} | R_sqr = {r2} | X_0 = {x_0}', # title of plot
        xaxis_title_text = 'Feature value', # xaxis label
        yaxis_title_text = 'WoE', # yaxis label
        height = 500
            )
    
    fig.show()
    
    if clip:
        hist = px.histogram(df_buckets[df_buckets[feature_name] <= np.percentile(df_buckets[feature_name], 95)].sort_values(by = 'bucket'), x=feature_name, color = 'bucket')
    else:
        hist = px.histogram(df_buckets.sort_values(by = 'bucket'), x=feature_name, color = 'bucket')
        
    hist.update_layout(
        title_text = f'Distribution of {feature_name}', # title of plot
        xaxis_title_text = 'Feature value', # xaxis label
        legend_title_text = 'Bucket number',
        height = 500
            )
    hist.show()
    
    
def bucket_line(X, y, n_buckets = 20, feature_name = 'feature', target_name = 'target', clip = True, alpha = 0.05, interactive = False):
    
    """ Строит график зависимости WoE
    x - параметр, от которого стоит искать зависимость
    y - метки класса (0 / 1)
    n_buckets - количество бинов для вещественного признака
    feature_name, target_name - подписи к графику
    """
    
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.stats import norm
    
    def make_bucket(feature, n=100):
        '''функция, бьющая на бакеты(по умолчанию 100 точек)'''
        return df.assign(bucket = np.ceil(feature.rank(pct=True) * n))
    
    def WoE (df_1, feature_name = 'feature', target_name = 'target'):

        a_1 = df_1[df_1 == 1].count()
        b_1 = df_1[df_1 == 0].count()

        a = df[df[target_name] == 1].count()[feature_name]
        b = df[df[target_name] == 0].count()[feature_name]

        c = np.log(a_1/b_1) - np.log(a/b)

        return c

    def p_diff(df_1, feature_name = 'feature', target_name = 'target'):

        a_1 = df_1[df_1 == 1].count()
        b_1 = df_1[df_1 == 0].count()

        a = df[df[target_name] == 1].count()[feature_name]
        b = df[df[target_name] == 0].count()[feature_name]

        c = a_1/a - b_1/b

        return c
     
        
    df = pd.DataFrame({feature_name : X[feature_name], target_name : y})
    df = df[df[feature_name].notna()]
    
    df_buckets = make_bucket(df[feature_name], n_buckets)
    df_info = df_buckets.groupby('bucket').agg(
            feature_value = pd.NamedAgg(column = feature_name, aggfunc='mean'),
            target_value =  pd.NamedAgg(column = target_name, aggfunc ='mean'),
            target_std = pd.NamedAgg(column = target_name, aggfunc ='std'),
            target_cnt = pd.NamedAgg(column = target_name, aggfunc ='count')
    )
    
    df_info['target_ci'] = norm.pdf(1 - alpha)*df_info['target_std']/np.sqrt(df_info['target_cnt'])

    fig = go.Figure(data=go.Scatter(
            name = 'Initial data',
            y=df_info['target_value'],
            x=df_info['feature_value'],
            line = (dict(
                color = 'indianred',
                dash = 'dot')),
            error_y=dict(
                type='data',
                array = df_info['target_ci'],
                symmetric=True,
                color = 'indianred')
            ))
    
    
    model = LinearRegression()

    X = np.array(df_info['feature_value']).reshape(-1, 1)
    y = np.array(df_info['target_value'])
    model.fit(X = X, y = y)
    x = np.linspace(np.min(df_info['feature_value']), np.max(df_info['feature_value']), 100)
    r2 = np.round(r2_score(df_info['target_value'], model.predict(np.array(df_info['feature_value']).reshape(-1,1))), 3)
    

    x_0 = np.round(- model.intercept_/model.coef_[0], 3)
    
    fig.add_trace(go.Scatter(
                name = 'Regression',
                x = x,
                y = model.predict(np.array(x).reshape(-1,1)),
                line = dict(
                color = 'blue'
                )))
    
    fig.update_layout(
        title_text = f' {feature_name} | R_sqr = {r2} | X_0 = {x_0}', # title of plot
        xaxis_title_text = feature_name, # xaxis label
        yaxis_title_text = target_name, # yaxis label
        height = 500
            )
    
    fig.show(config={'staticPlot': not interactive})
    
    if clip:
        hist = px.histogram(df_buckets[df_buckets[feature_name] <= np.percentile(df_buckets[feature_name], 95)].sort_values(by = 'bucket'), x=feature_name, color = 'bucket')
    else:
        hist = px.histogram(df_buckets.sort_values(by = 'bucket'), x=feature_name, color = 'bucket')
        
    hist.update_layout(
        title_text = f'Distribution of {feature_name}', # title of plot
        xaxis_title_text = feature_name, # xaxis label
        legend_title_text = 'Bucket number',
        height = 500
            )
    hist.show(config={'staticPlot': not interactive})
    
    
from sklearn.linear_model import LogisticRegression
from scipy.stats import f, chi2


def likelihood_ratio_test(ll_short, ll_long):
    
    """
    вспомогательная функция
    рассчитывает значение p-value для теста отношения правдоподобия
    ll_short — логарифм правдоподобия "короткой" модели
    ll_long — логарифм правдоподобия "длинной" модели

    Returns
    -----
    p-value
    """

    from scipy.stats.distributions import chi2

    lr = 2 * (ll_short - ll_long)
    return chi2.sf(lr, 1)

def  stepwise_selection_classification(df, features, model,  target='d4p12', alpha_in=0.05, alpha_out = 0.10):
    
    """
    Функция для отбора признаков при помощи прямого прохода 
    
    Parameters
    ----------
    df : DataFrame
        датафрейм с наблюдениями и целевой переменной
    val_size : str
        размер валидационной выборки
    test_size : str
        размер тестовой выборки
    alpha_in : float in range [0, 1]
        уровень значимости вхождения параметра в модель
    alpha_out : float in range [0, 1]
        уровень значимости выхода параметра из модели

    Returns
    ---
    selected_features : list
        список переменных отобранных на заданном уровне значимости alpha 
    """


    selected_features = list()
    n = df.shape[0]
    p_value = 1
    while True:
        fl_best = False
        potential_features = list(set(features) - set(selected_features))
        best_feature = ''
        for feature in potential_features:
            temp_features = [feature] + selected_features
            lr_short = model()
            lr_long = model()
            if len(selected_features) == 0:
                const = pd.Series([1]*df.shape[0])
                lr_short.fit(np.array(const).reshape(-1, 1), df[target])
                ll_short = log_loss(df[target], lr_short.predict_proba(np.array(const).reshape(-1, 1))[:, 1], normalize=False) 
                lr_long.fit(np.array(df[temp_features]).reshape(-1, 1), df[target])
                ll_long = log_loss(df[target], lr_long.predict_proba(np.array(df[temp_features]).reshape(-1, 1))[:, 1], normalize=False)
            else:
                lr_short.fit(df[selected_features], df[target])
                ll_short = log_loss(df[target], lr_short.predict_proba(df[selected_features])[:, 1], normalize=False)
                lr_long.fit(df[temp_features], df[target])
                ll_long = log_loss(df[target], lr_long.predict_proba(df[temp_features])[:, 1], normalize=False)
            if likelihood_ratio_test(ll_short, ll_long) < alpha_in and likelihood_ratio_test(ll_short, ll_long) < p_value:
                p_value = likelihood_ratio_test(ll_short, ll_long)
                best_feature = feature
        if best_feature == '':
            break
        else:
            selected_features.append(best_feature)
            print(f"В модель была добавлена переменная {best_feature}, p-value: {round(p_value, 4)}")
        
        p_value = 0
        fl_worst = False
        worst_feature = ''
        for feature in selected_features:
            temp_features = selected_features.copy()
            temp_features.remove(feature)
            lr_short = model()
            lr_long = model()
            if len(temp_features) == 0:
                const = pd.Series([1]*df.shape[0])
                lr_short.fit(np.array(const).reshape(-1, 1), df[target])
                ll_short = log_loss(df[target], lr_short.predict_proba(np.array(const).reshape(-1, 1))[:, 1], normalize=False) 
                lr_long.fit(np.array(df[selected_features]).reshape(-1, 1), df[target])
                ll_long = log_loss(df[target], lr_long.predict_proba(np.array(df[selected_features]).reshape(-1, 1))[:, 1], normalize=False)
            else:
                lr_short.fit(df[temp_features], df[target])
                ll_short = log_loss(df[target], lr_short.predict_proba(df[temp_features])[:, 1], normalize=False)
                lr_long.fit(df[selected_features], df[target])
                ll_long = log_loss(df[target], lr_long.predict_proba(df[selected_features])[:, 1], normalize=False)
            if likelihood_ratio_test(ll_short, ll_long) > alpha_out and likelihood_ratio_test(ll_short, ll_long) > p_value:
                p_value = likelihood_ratio_test(ll_short, ll_long)
                worst_feature = feature
        if worst_feature == '':
            fl_worst = True
        else:
            selected_features.remove(worst_feature)
            print(f"Из модели была исключена {worst_feature}, p-value: {round(p_value, 4)}")
        p_value = 1
    return selected_features

def sse(x_real, x_pred):
    
    return np.square((x_real - x_pred)).sum()

def f_test(sse_r, sse_u, n_obs, k_params, q_restr):
    
    f_stat = ((sse_r - sse_u)/q_restr)/(sse_u/(n_obs - k_params - 1))
    return f_stat, f.sf(f_stat, k_params-1, n_obs-k_params)


def forward_selection_regression(df, features, model, target='target', alpha_in=0.05, alpha_out = 0.10):
    
    """
    Функция для отбора признаков при помощи прямого прохода 
    
    Parameters
    ----------
    df : DataFrame
        датафрейм с наблюдениями и целевой переменной
    val_size : str
        размер валидационной выборки
    test_size : str
        размер тестовой выборки
    alpha_in : float in range [0, 1]
        уровень значимости вхождения параметра в модель
    alpha_out : float in range [0, 1]
        уровень значимости выхода параметра из модели

    Returns
    ---
    selected_features : list
        список переменных отобранных на заданном уровне значимости alpha 
    """


    selected_features = list()
    n = df.shape[0]
    p_value = 1
    while True:
        fl_best = False
        potential_features = list(set(features) - set(selected_features))
        best_feature = ''
        for feature in potential_features:
            temp_features = [feature] + selected_features
            model_short = model()
            model_long = model()
            n_obs = df.shape[0]
            k_params = len(temp_features) + 1
            q_restr = 1
            if len(selected_features) == 0:
                const = pd.Series([1]*df.shape[0])
                model_short.fit(np.array(const).reshape(-1, 1), df[target])
                sse_short = sse(df[target], model_short.predict(np.array(const).reshape(-1, 1))) 
                model_long.fit(np.array(df[temp_features]).reshape(-1, 1), df[target])
                sse_long = sse(df[target], model_long.predict(np.array(df[temp_features]).reshape(-1, 1)))
                p_value_ = f_test(sse_short, sse_long, n_obs, k_params, q_restr)[1]
            else:
                model_short.fit(df[selected_features], df[target])
                sse_short = sse(df[target], model_short.predict(df[selected_features]))
                model_long.fit(df[temp_features], df[target])
                sse_long = sse(df[target], model_long.predict(df[temp_features]))
                p_value_ = f_test(sse_short, sse_long, n_obs, k_params, q_restr)[1]
            if p_value_ < alpha_in and p_value_ < p_value:
                p_value = p_value_
                best_feature = feature
        if best_feature == '':
            break
        else:
            selected_features.append(best_feature)
            print(f"В модель была добавлена переменная {best_feature}, p-value: {round(p_value, 4)}")

        p_value = 1
    return selected_features