import pickle as pk
import pandas

class DecisionTreeModel:
    def __init__(self):
        self.model_pipe = pk.load(open('./pickles/decision_tree_pipe_spotify.pkl', 'rb'))
        self.scaler = {
            "z_core": pk.load(open('./pickles/encoder_z_score.pkl', 'rb')),
            "min_max": pk.load(open('./pickles/encoder_min_max.pkl', 'rb')),
            "one_hot": pk.load(open('./pickles/encoder_one_hot.pkl', 'rb')),
            "ta_encoder": pk.load(open('./pickles/encoder_ta.pkl', 'rb'))

        }

        self.model_manual = pk.load(open("./pickles/decision_tree.pkl", "rb"))

    def predict(self, df):
        return self.model_pipe.predict(df)
