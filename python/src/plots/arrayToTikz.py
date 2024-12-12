import numpy as np
import pandas as pd

# Output like (0,0) -- (0.5, 0) -- (1, 1);
def arrayToTikz(xArray: np.ndarray, yArray: np.ndarray, roundNum=4) -> str:
    if len(xArray) != len(yArray):
        raise ValueError(f'Arrays dont have the same size: {xArray.size} and {yArray.size}')
    output = ""
    for i in range(len(xArray)):
        if i > 0:
            output += " -- \n"
        output += f"({np.round(xArray[i], 4)},{np.round(yArray[i], roundNum)})"
    output += ";"
    return output

def arrayToCsv(xArray, yArray, filename='data'):
    df = pd.DataFrame()
    df['x'] = xArray.tolist()
    df['value'] = yArray.tolist()
    df.to_csv(f'{filename}.csv', index=False)

def main():
    xArray = np.linspace(start=0, stop=1, num=10)
    yArray = np.sin(xArray)
    arrayToCsv(xArray, yArray)
    print(xArray)
    print(yArray)
    #print(arrayToTikz(xArray, yArray, 10))

if __name__ == "__main__":
    main()