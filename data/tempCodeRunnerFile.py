    import pandas
    dir_path = os.path.dirname(__file__)
    dataset_up_path = os.path.join(dir_path, "balloon/datasets_up.csv")
    dataset_down_path = os.path.join(dir_path, "balloon/datasets_down.csv")

    df =  pandas.read_csv(dataset_up_path, delimiter=',')
    print(df)