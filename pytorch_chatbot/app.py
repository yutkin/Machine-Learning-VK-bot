import torch
import torch.nn as nn
import os

import flask
from flask import Flask

import vocab
import encoder
import decoder
import utils
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] PID: %(process)d %(levelname)s @ "
    "%(pathname)s:%(lineno)d ~ %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)

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
            flask.current_app.traced_encoder,
            flask.current_app.traced_decoder,
            flask.current_app.scripted_searcher,
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
    device = torch.device("cpu")

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

    loadFilename = "/app/8000_checkpoint.tar"

    # Load model
    checkpoint = torch.load(loadFilename, map_location=device)
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
    enc = encoder.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    dec = decoder.LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
    )
    # Load trained model params
    enc.load_state_dict(encoder_sd)
    dec.load_state_dict(decoder_sd)
    # Use appropriate device
    enc = enc.to(device)
    dec = dec.to(device)
    # Set dropout layers to eval mode
    enc.eval()
    dec.eval()

    # Convert encoder model
    # Create artificial inputs
    test_seq = torch.LongTensor(10, 1).random_(0, voc.num_words)
    test_seq_length = torch.LongTensor([test_seq.size()[0]])
    # Trace the model
    traced_encoder = torch.jit.trace(enc, (test_seq, test_seq_length))

    # Convert decoder model
    # Create and generate artificial inputs
    test_encoder_outputs, test_encoder_hidden = traced_encoder(
        test_seq, test_seq_length
    )
    test_decoder_hidden = test_encoder_hidden[: dec.n_layers]
    test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
    # Trace the model
    traced_decoder = torch.jit.trace(
        dec, (test_decoder_input, test_decoder_hidden, test_encoder_outputs)
    )

    # Initialize searcher module
    scripted_searcher = decoder.GreedySearchDecoder(
        traced_encoder, traced_decoder, dec.n_layers
    )

    app.traced_encoder = traced_encoder
    app.traced_decoder = traced_decoder
    app.scripted_searcher = scripted_searcher
    app.voc = voc

    app.run(host="0.0.0.0", port=80, debug=False)
