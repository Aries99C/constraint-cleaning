import pandas as pd


if __name__ == '__main__':
    idf = pd.read_csv('./data/IDF.csv')
    idf.drop(['timestamp'], axis=1, inplace=True)

    idf = idf[:200]

    for col in idf.columns:
        idf[col] = idf[col] * 100
        idf[col] = idf[col].astype(int)

    with open('idf_data.txt', 'w') as f:
        for line in idf.values:
            for i in range(len(line)):
                f.write(str(line[i]))
                if i < len(line) - 1:
                    f.write(',')
                else:
                    f.write('\n')
