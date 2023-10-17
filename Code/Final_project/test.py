import pandas as pd

# Create a sample DataFrame
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print("DataFrame 1:")
print(df1)

# Create another sample DataFrame
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
print("\nDataFrame 2:")
print(df2)

# Append df2 to df1
df1 = df1.append(df2, ignore_index=True)
print("\nAppended DataFrame:")
print(df1)
