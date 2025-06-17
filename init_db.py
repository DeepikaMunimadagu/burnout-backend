import sqlite3

conn = sqlite3.connect("burnout.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    gender INTEGER,
    company_type INTEGER,
    wfh INTEGER,
    designation INTEGER,
    resource_allocation REAL,
    mental_fatigue_score REAL,
    burnout_risk INTEGER,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print("Database initialized ✅")
