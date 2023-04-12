import sqlite3
from flask import Flask,render_template,Response

app=Flask(__name__)
connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO db1 (anomaly_type, time_of_occur) VALUES ('/static/img3.jpg', 'FIGHT')")
# cur.execute("INSERT INTO db1 (title, content) VALUES (?, ?)",
#             ('First Post', 'Content for the first post')
#             )

# cur.execute("INSERT INTO posts (title, content) VALUES (?, ?)",
#             ('Second Post', 'Content for the second post')
#             )


connection.commit()
connection.close()

if __name__ == "__main__":
    app.run(debug=True)