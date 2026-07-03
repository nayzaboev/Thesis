import wbgapi as wb
import pandas as pd

countries = ['ARM','AZE','BLR','EST','GEO','KAZ','KGZ','LVA','LTU','MDA','RUS','TJK','UKR','UZB']
names = {'ARM':'Armenia','AZE':'Azerbaijan','BLR':'Belarus','EST':'Estonia',
         'GEO':'Georgia','KAZ':'Kazakhstan','KGZ':'Kyrgyzstan','LVA':'Latvia',
         'LTU':'Lithuania','MDA':'Moldova','RUS':'Russia','TJK':'Tajikistan',
         'UKR':'Ukraine','UZB':'Uzbekistan'}

# What we currently have in cultural_vars.csv (UN World Marriage Data 2019 latest)
current = {
    'Armenia': (24.1, 2015), 'Azerbaijan': (24.2, 2018), 'Belarus': (22.5, 2012),
    'Estonia': (33.6, 2018), 'Georgia': (22.8, 2014), 'Kazakhstan': (22.4, 2015),
    'Kyrgyzstan': (21.2, 2014), 'Latvia': (30.4, 2018), 'Lithuania': (32.7, 2018),
    'Moldova': (25.0, 2014), 'Russia': (24.4, 2010), 'Tajikistan': (20.7, 2017),
    'Ukraine': (23.0, 2012), 'Uzbekistan': (22.5, 2006),
}

df = wb.data.DataFrame('SP.DYN.SMAM.FE', economy=countries, time=range(1990,2024),
                       labels=False)
df = df.reset_index().melt(id_vars='economy', var_name='year', value_name='smam')
df['year'] = df['year'].str.replace('YR','',regex=False).astype(int)
df['country'] = df['economy'].map(names)
df = df.dropna(subset=['smam']).sort_values(['country','year'])

print("=== World Bank SP.DYN.SMAM.FE — all non-missing values per country ===\n")
for c in sorted(df['country'].unique()):
    sub = df[df['country']==c]
    yrs = [(int(y), round(v,2)) for y,v in zip(sub['year'], sub['smam'])]
    cur_val, cur_yr = current.get(c, ('?','?'))
    latest_wb = yrs[-1] if yrs else None
    flag = ""
    if latest_wb and latest_wb[0] > cur_yr:
        flag = f"  *** NEWER than current ({cur_val} @ {cur_yr}) — CONSIDER UPGRADE"
    elif latest_wb and latest_wb[0] == cur_yr:
        flag = f"  same year as current"
    else:
        flag = f"  older than current ({cur_val} @ {cur_yr}) — keep current"
    print(f"{c}: {yrs}{flag}")

print("\nDone. Newer years marked with *** should be considered for upgrade.")