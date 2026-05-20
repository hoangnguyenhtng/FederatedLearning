import pandas as pd

df = pd.read_pickle('data/processed/multi_category/client_0/data.pkl')

print(type(df['image_embedding'].iloc[0]))
print(df['image_embedding'].iloc[0][:5])  # xem thử vài phần tử đầu
print(len(df['image_embedding'].iloc[0])) # độ dài embedding