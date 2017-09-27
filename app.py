from flask import Flask, render_template, jsonify, request
import app_bot

Bot = app_bot.ShakespeareBot()

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('chat.html')

'''
@app.route('/hello')
def hello_world():
    return 'Greetings from the 17th century.'
    '''

@app.route('/ask', methods=['GET', 'POST'])
def server():
    text = str(request.form['messageText'])

    if text == 'quit':
        exit()
    else:
        response = Bot.respond(text)

    return jsonify({'status': 'OK', 'answer': response})

if __name__ == '__main__':
    app.run()
