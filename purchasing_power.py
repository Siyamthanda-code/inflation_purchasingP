import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. DATA LOADING (Same cleaning as always)
# ==========================================

raw_data = """
Year	Jan	Feb	Mar	Apr	May	Jun	Jul	Aug	Sep	Oct	Nov	Dec	Average
1975	15,2	14,7	13,7	14,6	14,8	14,2	13,7	13,0	12,2	12,2	12,[1]	11,7	13,5
1976	11,4	10,9	11,5	11,3	11,6	11,1	10,9	11,2	11,4	11,1	10,4	10,8	11,1
1977	10,7	11,9	11,8	11,5	11,1	11,0	11,3	11,3	11,3	11,2	11,4	11,1	11,3
1978	11,3	10,6	9,9	9,7	9,5	9,2	12,5	11,7	11,7	11,7	11,5	11,6	10,9
1979	11,6	11,3	12,6	12,8	12,8	13,5	12,9	13,9	14,3	14,2	14,2	14,0	13,2
1980	13,8	14,3	13,1	13,4	14,1	14,6	12,2	11,9	12,8	14,4	14,9	15,8	13,8
1981	15,5	16,0	16,2	15,5	15,0	14,5	15,5	16,1	15,6	14,5	14,4	13,9	15,2
1982	13,9	13,6	15,4	16,5	16,5	16,1	14,4	13,8	14,1	14,3	14,2	13,8	14,7
1983	14,4	14,9	13,6	12,6	12,8	12,4	12,1	12,2	10,9	10,7	10,6	11,0	12,4
1984	10,3	10,0	10,2	11,0	11,0	11,7	12,4	11,8	12,2	12,5	13,3	13,3	11,6
1985	13,3	16,0	15,1	15,8	16,1	16,4	15,9	16,4	16,6	16,8	16,9	18,4	16,1
1986	20,7	18,1	18,9	18,6	17,5	16,9	18,2	18,7	19,7	19,2	19,2	18,1	18,7
1987	16,1	16,3	16,8	16,2	17,3	17,2	16,3	16,3	15,8	15,8	15,0	14,7	16,1
1988	14,2	13,7	13,4	13,3	12,9	12,4	12,4	12,3	12,4	12,3	12,4	12,5	12,9
1989	13,3	13,5	13,8	14,0	14,9	15,7	13,5	15,5	14,9	14,8	14,9	15,3	14,7
1990	15,1	14,9	14,9	14,6	13,9	13,6	13,3	13,6	14,3	14,0	15,3	14,6	14,4
1991	14,3	15,0	15,7	15,6	15,2	15,2	15,8	15,6	15,4	16,8	15,5	16,2	15,3
1992	15,8	15,8	15,7	15,6	14,8	15,1	14,6	14,3	13,5	11,7	11,0	9,6	13,9
1993	9,7	9,0	9,7	11,0	10,6	10,0	9,9	9,3	9,1	9,4	9,2	9,5	9,7
1994	9,9	9,9	9,0	7,1	7,2	7,5	8,2	9,4	10,1	9,8	9,9	9,9	9,0
1995	9,6	9,9	10,2	11,0	10,8	10,0	9,0	7,5	6,4	6,3	6,4	6,9	8,7
1996	6,9	6,5	6,3	5,5	5,9	6,9	7,1	7,5	8,4	9,1	9,2	9,4	7,4
1997	9,4	9,8	9,6	9,9	9,5	8,8	9,1	8,7	8,0	7,5	6,8	6,1	8,6
1998	5,6	5,4	5,4	5,0	5,1	5,2	6,6	7,6	9,1	9,0	9,4	9,0	6,9
1999	8,9	8,6	7,9	7,7	7,1	7,3	4,9	3,2	1,9	1,7	1,9	2,2	5,1
2000	2,6	2,4	3,4	4,6	5,1	5,1	5,9	6,8	6,8	7,1	7,1	7,0	5,3
2001	7,1	7,8	7,4	6,5	6,4	6,3	5,3	4,6	4,4	4,0	4,3	4,6	5,7
2002	5,0	5,9	6,2	7,4	7,8	8,0	9,6	10,4	11,2	13,0	12,9	12,4	9,2
2003	11,6	10,3	10,2	8,8	7,8	6,7	5,2	5,1	3,7	1,5	0,4	0,3	5,8
2004	0,2	0,7	0,4	0,2	0,6	1,2	1,6	1,0	1,3	2,4	3,7	3,4	1,4
2005	3,0	2,6	3,0	3,4	3,3	2,8	3,4	3,9	4,4	4,0	3,4	3,6	3,4
2006	4,0	3,9	3,4	3,3	3,9	4,9	5,0	5,4	5,3	5,4	5,4	5,8	4,7
2007	6,0	5,7	6,1	7,0	6,9	7,0	7,0	6,7	7,2	7,9	8,4	9,0	7,1
2008	9,3	9,8	10,6	11,1	11,7	12,2	13,4	13,7	13,1	12,1	11,8	9,5	11,5
2009	8,1	8,6	8,5	8,4	8,0	6,9	6,7	6,4	6,1	5,9	5,8	6,3	7,1
2010	6,2	5,7	5,1	4,8	4,6	4,1	3,7	3,5	3,2	3,4	3,6	3,5	4,3
2011	3,7	3,7	4,1	4,2	4,6	5,0	5,3	5,3	5,7	6,0	6,1	6,1	5,0
2012	6,3	6,1	6,0	6,1	5,7	5,5	4,9	5,0	5,5	5,6	5,6		5,7	5,6
2013	5,4	5,9	5,9	5,9	5,6	5,5	6,3	6,4	6,0	5,5	5,3	5,4	5,7
2014	5,8	5,9	6,0	6,1	6,6	6,6	6,3	6,4	5,9	5,9	5,8	5,3	6,1
2015	4,4	3,9	4,0	4,5	4,6	4,7	5,0	4,6	4,6	4,7	4,8	5,2	4,6
2016	6,2	7,0	6,3	6,2	6,1	6,3	6,0	5,9	6,1	6,4	6,6	6,8	6,4
2017	6,6	6,3	6,1	5,3	5,4	5,1	4,6	4,8	5,1	4,8	4,6	4,7	5,3
2018	4,4	4,0	3,8	4,5	4,4	4,6	5,1	4,9	4,9	5,1	5,2	4,5	4,7
2019	4,0	4,1	4,5	4,4	4,5	4,5	4,0	4,3	4,1	3,7	3,6	4,0	4,1
2020	4,5	4,6	4,1	3,0	2,1	2,2	3,2	3,1	3,0	3,3	3,2	3,1	3,3
2021	3,2	2,9	3,2	4,4	5,2	4,9	4,6	4,9	5,0	5,0	5,5	5,9	4,5
2022	5,7	5,7	5,9	5,9	6,5	7,4	7,8	7,6	7,5	7,6	7,4	7,2	6,9
2023	6,9	7,0	7,1	6,8	6,3	5,4	4,7	4,8	5,4	5,9	5,5	5,1	6,0
2024	5,3	5,6	5,3	5,2	5,2	5,1	4,6	4,4	3,8	2,8	2,9	3,0	4,4
2025	3,2	3,2	2,7	2,8	2,8	3,0	3,5	3,3	3,4	3,6	3,5	3,6	3,2
2026	3,5	3,0
"""

# --- CLEANING ---
clean_data = raw_data.replace('[1]', '1')
lines = clean_data.strip().split('\n')
header = lines[0].split('\t')
expected_cols = len(header)
processed_lines = [header] 
for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) < expected_cols:
        parts += [''] * (expected_cols - len(parts))
    if len(parts) > expected_cols:
        parts = parts[:expected_cols]
    processed_lines.append(parts)

clean_csv_string = "\n".join(["\t".join(row) for row in processed_lines])
df = pd.read_csv(io.StringIO(clean_csv_string), sep='\t')

# Convert numeric strings
cols_to_convert = df.columns.drop('Year')
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '.').replace('', np.nan).astype(float)

# Reshape to Long Format
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

df_long = df.melt(id_vars=['Year'], var_name='Month', value_name='Inflation_Rate')
df_long['Month'] = df_long['Month'].map(month_map)
df_long['Date'] = pd.to_datetime(df_long[['Year', 'Month']].assign(DAY=1))
df_long = df_long.set_index('Date').sort_index()
df_long = df_long[~df_long.index.duplicated(keep='first')]
df_long['Inflation_Rate'] = df_long['Inflation_Rate'].ffill()
df_long = df_long.asfreq('MS')

# ==========================================
# 2. DATA AUGMENTATION: THE SALARY SIMULATION
# ==========================================
print("Simulating Worker Wages and Calculating Real Purchasing Power...")

# Scenario: A worker starts at R10,000/month in 2000 and gets a standard 5% raise every year.
# We focus on 2000-2026 for this analysis.
start_date = '2000-01-01'
end_date = '2026-12-31'
df_sim = df_long.loc[start_date:end_date].copy()

# Calculate Monthly Inflation Factor (e.g., 5% inflation -> 1.05)
df_sim['Inflation_Factor'] = 1 + (df_sim['Inflation_Rate'] / 100)

# Calculate Nominal Salary Growth (5% per year = ~0.4% per month)
annual_raise = 0.05
monthly_raise = (1 + annual_raise) ** (1/12) - 1
df_sim['Monthly_Raise_Factor'] = 1 + monthly_raise

# 1. Nominal Salary (What is written on the payslip)
# Compounding the monthly raises
df_sim['Nominal_Salary'] = 10000 * df_sim['Monthly_Raise_Factor'].cumprod()

# 2. Cumulative Inflation Index (How much R1 is worth today vs 2000)
df_sim['CPI_Index'] = df_sim['Inflation_Factor'].cumprod()

# 3. Real Salary (Purchasing Power)
# Formula: Nominal / Inflation_Index
# This tells us: "How much is today's salary worth in 2000 Rands?"
df_sim['Real_Salary_2000_Rands'] = df_sim['Nominal_Salary'] / df_sim['CPI_Index']

# Calculate percentage loss of purchasing power
initial_real = df_sim['Real_Salary_2000_Rands'].iloc[0]
df_sim['Purchasing_Power_Change'] = ((df_sim['Real_Salary_2000_Rands'] - initial_real) / initial_real) * 100

print(f"Starting Nominal Salary (2000): R{df_sim['Nominal_Salary'].iloc[0]:,.2f}")
print(f"Ending Nominal Salary (2026): R{df_sim['Nominal_Salary'].iloc[-1]:,.2f}")
print(f"Starting Real Salary (2000 value): R{df_sim['Real_Salary_2000_Rands'].iloc[0]:,.2f}")
print(f"Ending Real Salary (2000 value): R{df_sim['Real_Salary_2000_Rands'].iloc[-1]:,.2f}")
print(f"Total Purchasing Power Change: {df_sim['Purchasing_Power_Change'].iloc[-1]:.2f}%")

# ==========================================
# 3. VISUALIZATION: THE SILENT EROSION
# ==========================================
plt.figure(figsize=(14, 7))

# Plot Nominal Salary (Green - going up, looks good)
plt.plot(df_sim.index, df_sim['Nominal_Salary'], label='Nominal Salary (Face Value)', color='green', linewidth=2)

# Plot Real Salary (Red - actual value)
plt.plot(df_sim.index, df_sim['Real_Salary_2000_Rands'], label='Real Salary (Purchasing Power - 2000 Rands)', color='red', linewidth=2, linestyle='--')

# Formatting
plt.title('The Silent Erosion: Salary Growth vs. Inflation (2000-2026)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Salary Value (Rands)')
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight the gap
plt.fill_between(df_sim.index, df_sim['Real_Salary_2000_Rands'], df_sim['Nominal_Salary'], 
                 color='orange', alpha=0.1, label='Inflation Gap')

plt.tight_layout()
plt.savefig('purchasing_power_analysis.png', dpi=300)
plt.show()

print("\n✅ Analysis Complete. Check 'purchasing_power_analysis.png'.")