import fasttext
import flask
import numpy as np
import gensim


app = flask.Flask('API')
#fasttext_model_path = 'data/cc.en.50.bin'
#ft_model = fasttext.load_model(fasttext_model_path)

@app.route('/')
def heartbeat():
    return flask.jsonify({'alive':True})

@app.route('/math', methods = ['GET'])
def do_math():
    number = int(flask.request.args.get('number'))
    return flask.jsonify({'status': 'complete', 'number': number * 10})

@app.route('/thesaurus')
def thes():
    return flask.render_template('thesaurus.html')

@app.route('/sentence', methods='[GET]')
def sentence():
    input = flask.request.args.get('sentence')
    if (input and len(input) > 0):
        input = [v.lower() for v in input_sentence.split('_')]
        model_vector = [gen_model.get_vector(v) for v in input]
        av_vector = np.mean(model_vector, axis = 0)
        similar = gen_model.similar_by_vector(av_vector)
        return flask.jsonify({'similar':similar, 'success': True})
if __name__  == '__main__':
    app.run(port=8000)