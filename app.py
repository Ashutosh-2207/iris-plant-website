from flask import Flask
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    file_path = r"C:\Users\ASUS\Desktop\iris-plant-website\iris.csv"
    if not os.path.exists(file_path):
        return "<h2 style='color:red;'>CSV file not found at:<br>" + file_path + "</h2>"

    # Read CSV
    df = pd.read_csv(file_path)

    # Rename columns
    df.rename(columns={
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm': 'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm': 'petal_width',
        'Species': 'species'
    }, inplace=True)

    # Group by species and calculate means
    grouped_means = df.groupby('species')[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()

    # Convert to HTML table with Bootstrap classes
    html_table = grouped_means.to_html(classes='table table-striped table-bordered')

    # Return a simple HTML page with the table
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Dataset Averages</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                margin: 20px;
                background-color: #f8f9fa;
            }}
            h1 {{
                margin-bottom: 30px;
            }}
            .container {{
                max-width: 700px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Average Iris Feature Values by Species</h1>
            {html_table}
        </div>
    </body>
    </html>
    """

    return html

if __name__ == '__main__':
    app.run(debug=True)
