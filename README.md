# Dialogue bot for Early Modern English

Playing with a seq2seq style chatbot trained on Early Modern English (ie., a Shakespeare chatbot)

## System Notes

Python 3.5

## About the project

I put this project together mostly as a training exercise for myself, to go through the procedure of creating a simple chatbot, and also to test out the simple hypothesis that if you train a system with stylized language, the results will be in that style as well. Sounds obvious, but stylized language generation has other applications, so it's nice to confirm that it works.

## How to run the app/test

Locally, from command-line:

`python3 app.py`

It will load up, and prompt you that the chatbot is available at your local address. Copy & paste the address into the browser and chat away.
It is very silly.

## How to run at command-line

Locally, from command-line:
`python3 dialogue.py --decode=True --train_dir=./location/of/training/dir --data_dir=./location/of/data/dir`

It will load up and give you prompts at the command line to interact with the bot.