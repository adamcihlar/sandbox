''' This file does basically nothing '''
import pandas as pd
import sys

df = pd.DataFrame({"a":[1,2,3,4], "b":[3,4,5,6]})
if sys.argv[1]=='print':
    print(df)
