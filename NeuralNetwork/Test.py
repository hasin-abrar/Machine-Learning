x_array = np.array(df['total_bedrooms'])
normalized_X = preprocessing.normalize([x_array])

x_df = pd.DataFrame(x)
    names = x_df.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(x_df)
    x_stan_df = pd.DataFrame(scaled_df, columns=names)
    x = x_stan_df.values