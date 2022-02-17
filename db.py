import pymssql

conn = pymssql.connect('172.30.1.50', 'bmeks', 'qlaprtm1@', 'T_TIMS_1_RE')
cursor = conn.cursor()

name = 'E-'
cursor.execute(f"SELECT Main_Name, Tube_cnt FROM ENERGY.SEPIP_Mater_Thick WHERE Main_Name like '{name}%';")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()