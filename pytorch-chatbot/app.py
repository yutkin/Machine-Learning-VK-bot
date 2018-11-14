import torch
import torch.nn as nn
import os

import flask
from flask import Flask

import vocab
import models
import utils
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] PID: %(process)d %(levelname)s @ "
    "%(pathname)s:%(lineno)d ~ %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/reply", methods=["POST"])
def bot_reply():
    try:
        data = flask.request.json
        msg = data["message"]
    except (ValueError, KeyError):
        return flask.abort(400)

    try:
        reply = utils.reply_on_sentence(
            msg,
            flask.current_app.encoder,
            flask.current_app.decoder,
            flask.current_app.searcher,
            flask.current_app.voc,
        )
    except KeyError:
        reply = "Sorry, I didn't catch that..."
        logger.exception(reply)
    except Exception:
        reply = "Oops, something bad is going on..."
        logger.exception(reply)

    return flask.jsonify({"reply": reply.replace(" .", ".").capitalize()})


if __name__ == "__main__":
    save_dir = os.path.join("data", "save")
    corpus_name = "cornell movie-dialogs corpus"

    # Configure models
    model_name = "cb_model"
    attn_model = "dot"
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    loadFilename = "/app/10000_checkpoint.tar"

    # Load model
    if USE_CUDA:
        checkpoint = torch.load(loadFilename)
    else:
        checkpoint = torch.load(loadFilename, map_location=torch.device("cpu"))

    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]
    voc = vocab.Voc(corpus_name)
    voc.__dict__ = checkpoint["voc_dict"]

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = models.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = models.LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
    )
    # Load trained model params
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    searcher = models.GreedySearchDecoder(encoder, decoder)

    app.encoder = encoder
    app.decoder = decoder
    app.searcher = searcher
    app.voc = voc

    app.run(host="0.0.0.0", port=80, debug=False)
