from flask import Flask, render_template

# Khởi tạo Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """
    Trang chủ hiển thị index.html.
    """
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6868, debug=True)
